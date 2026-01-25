import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from imblearn.under_sampling import RandomUnderSampler
from losses import AsymmetricLossOptimized
from sklearn.metrics import classification_report, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from training_utils import (
    DummyLogger,
    get_examples_per_class,
    init_logging,
    open_json,
    plot_confusion_matrix,
    write_json,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logger = init_logging(logger_name="task_training", log_level=logging.DEBUG)


class LogTrainingMetricsCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = AsymmetricLossOptimized()

    def compute_loss(self, model, inputs, return_outputs=False):
        # By default Trainer will place the labels inside inputs under the "labels" key.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def process_dataset(
    data,
    tokenizer: AutoTokenizer,
    label_column: str,
    text_column: str,
    num_classes: int,
):
    label = data[label_column]
    labels = np.zeros(num_classes)
    if label is not None:
        label = int(label)
        labels[label] = 1

    data = tokenizer(
        data[text_column], truncation=True, padding="max_length", max_length=512
    )

    data["labels"] = labels
    return data


def get_performance_metrics(preds: EvalPrediction) -> Dict[str, float]:
    """Custom metrics function for the Trainer.

    Args:
        preds (EvalPrediction): Used internally. More info here: https://huggingface.co/docs/transformers/internal/trainer_utils#transformers.EvalPrediction

    Returns:
        Dict[str, float]: The computed metrics.
    """
    logits, labels = preds

    logits = F.sigmoid(torch.from_numpy(logits))
    logits = torch.where(logits > 0.5, 1, 0)

    f1_macro = f1_score(
        y_true=labels, y_pred=logits, average="macro"
    )  # use macro because our dataset is imbalanced.

    return {"f1_macro": f1_macro}


def multilabel_to_multiclass(
    labels: torch.Tensor, value_for_other_class: int
) -> torch.Tensor:
    # Find rows filled only with zeros.
    zero_rows = torch.all(labels == 0, axis=1)

    # Find the class.
    labels = labels.argmax(axis=1)

    # Add a specific value for the "other" class.
    # NOTE: Remember a zero vector represents our "other" class.
    labels[zero_rows] = value_for_other_class

    return labels


def freeze_or_unfreeze_layers(
    model: AutoModelForSequenceClassification, layers: List[str], freeze: bool = True
) -> AutoModelForSequenceClassification:
    """
    Freezes or unfreezes layers.

    Args:
        `layers` (List[str]):
            Which layers to affect.
        `freeze` (bool, optional):
            `False`: If you want to unfreeze the layers
            `True`: If you want to freeze the layers. Defaults to True.
    """
    for name, param in model.named_parameters():
        logger.debug(f"Layer: {name}, requires_grad: {param.requires_grad}")

        # If a layer requires_grad and we want to freeze it, freeze variable will be True.
        # So both of them will be True and we will change the requires_grad parameter to not freeze (= False).
        # If requires_grad and we want to unfreeze the layer, freeze will be False.
        # In that case, we won't do anything because the layer requires_grad already.
        if (
            any(layer.lower() in name.lower() for layer in layers)
            and param.requires_grad == freeze
        ):
            param.requires_grad = not freeze
            logger.debug(f"Layer: {name}, requires_grad: {param.requires_grad}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths and variables.
    this_file_path = Path(__file__).resolve().parent

    data_dir_path = os.path.join(this_file_path, "..", "data")
    splits_dir_path = os.path.join(data_dir_path, "splits")
    processed_dir_path = os.path.join(data_dir_path, "processed")

    inference_path = os.path.join(this_file_path, "..", "inference")
    config = open_json(filepath=os.path.join(this_file_path, "task_config.json"))

    # As a base model you can use: "nlpaueb/bert-base-greek-uncased-v1"
    if config["comes_from_phase1"]:
        base_model_path = os.path.join(
            inference_path,
            "contrastive_learning",
            config["task"],
            config["base_model_name"],
        )
    else:
        base_model_path = config["base_model_name"]

    model_id = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    model_path = os.path.join(
        inference_path,
        "task",
        config["task"],
        config["base_model_name"],
        f"{model_id}",
    )

    label2id = open_json(os.path.join(processed_dir_path, "label2id.json"))
    id2label = open_json(os.path.join(processed_dir_path, "id2label.json"))
    label_column = "label"
    text_column = "processed_text"

    # Load the datasets
    training_df: pd.DataFrame = pd.read_pickle(
        os.path.join(splits_dir_path, "train.pkl")
    )
    cl_validation_df: pd.DataFrame = pd.read_pickle(
        os.path.join(splits_dir_path, "cl_validation.pkl")
    )  # Load the cl val dataset and add it to the training set.
    test_df: pd.DataFrame = pd.read_pickle(os.path.join(splits_dir_path, "test.pkl"))
    validation_df: pd.DataFrame = pd.read_pickle(
        os.path.join(splits_dir_path, "validation.pkl")
    )
    training_df = pd.concat([training_df, cl_validation_df])
    training_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    validation_df.reset_index(inplace=True, drop=True)
    cl_validation_df.reset_index(inplace=True, drop=True)

    logger.debug(
        f"Training set Distribution:\n{training_df[label_column].value_counts(dropna=False)}"
    )
    logger.debug(
        f"Validation set Distribution:\n{validation_df[label_column].value_counts(dropna=False)}"
    )

    if config["use_undersampler"]:
        # Need to replace NaNs for it to work.
        none_class = training_df[label_column].max() + 1
        training_df[label_column].fillna(none_class, inplace=True)
        training_ratio = get_examples_per_class(
            df=training_df,
            label_column=label_column,
            diff_from_second=config["diff_from_second_class"],
        )
        rus = RandomUnderSampler(random_state=0, sampling_strategy=training_ratio)
        training_df, _ = rus.fit_resample(training_df, training_df[label_column])

        training_df.loc[training_df[label_column] == none_class, label_column] = None
        logger.debug(
            f"Training set Distribution:\n{training_df[label_column].value_counts(dropna=False)}"
        )

    config["training_examples"] = training_df.shape[0]
    config["validation_examples"] = validation_df.shape[0]
    config["test_examples"] = test_df.shape[0]
    # We need to drop the following columns from our datasets objects.
    remove_columns = training_df.columns.tolist()
    logger.debug(f"Remove columns: {remove_columns}")

    # Init training tracking.
    if config["enable_wandb_logging"]:
        wandb_config = open_json(filepath=os.path.join(this_file_path, "wandb.json"))
        wandb.login(key=wandb_config["key"])
        run = wandb.init(
            entity=wandb_config[
                "entity"
            ],  # NOTE: You can find your own under: User settings > Project Defaults.
            # Set the project where this run will be logged
            project=wandb_config["project"],
            # Track hyperparameters and run metadata
            config=config,
            name=model_id,
            # magic=True,
            job_type="Classification",
        )
    else:
        run = DummyLogger()

    logger.info(
        f"Training set Distribution:\n{training_df[label_column].value_counts(dropna=False)}",
    )
    logger.info(
        f"Validation set Distribution:\n{validation_df[label_column].value_counts(dropna=False)}",
    )
    logger.info(
        f"Test set Distribution:\n{test_df[label_column].value_counts(dropna=False)}"
    )

    training_dataset = Dataset.from_pandas(df=training_df)
    test_dataset = Dataset.from_pandas(df=test_df)
    validation_dataset = Dataset.from_pandas(df=validation_df)

    logger.debug(test_df.head())

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    training_dataset = training_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_column": label_column,
            "text_column": text_column,
            "num_classes": max(list(label2id.values())) + 1,
        },
        remove_columns=remove_columns,
    )
    validation_dataset = validation_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_column": label_column,
            "text_column": text_column,
            "num_classes": max(list(label2id.values())) + 1,
        },
        remove_columns=remove_columns,
    )
    test_dataset = test_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_column": label_column,
            "text_column": text_column,
            "num_classes": max(list(label2id.values())) + 1,
        },
        remove_columns=remove_columns,
    )
    logger.debug(training_dataset["labels"][:3])

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=max(list(label2id.values())) + 1,
        label2id=label2id,  # NOTE: If they contain Greek chars they will break the wandb logging!
        id2label=id2label,  # NOTE: If they contain Greek chars they will break the wandb logging!
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model = freeze_or_unfreeze_layers(model=model, layers=["embeddings"], freeze=True)

    logger.info(f"Run id: {run.id}")

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        greater_is_better=True,
        save_total_limit=2,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to="wandb" if config["enable_wandb_logging"] else "none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=get_performance_metrics,
    )

    # NOTE: We can uncomment the following line to log the metrics of the training set as well.
    # It makes the training process longer.
    trainer.add_callback(LogTrainingMetricsCallback(trainer))

    trainer.train()

    # Save the best model and its tokenizer to load them together.
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    write_json(data=label2id, filepath=os.path.join(model_path, "label2id.json"))
    write_json(data=id2label, filepath=os.path.join(model_path, "id2label.json"))

    # Get the Confusion Matrix.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    y_true = []
    y_pred = []

    # Convert the Dataset to tensors.
    test_dataset.set_format(type="torch")
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    value_for_other_class = max(label2id.values()) + 1

    with torch.no_grad():
        for inputs in tqdm(test_dataloader, desc="Parsing Test set"):
            labels = inputs.pop("labels")

            # labels = labels.argmax(axis=1)
            labels = multilabel_to_multiclass(
                labels=labels, value_for_other_class=value_for_other_class
            )  # Convert them to multiclass. E.g. [1, 0, 0, 0] -> 0

            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)

            logits = outputs.logits
            logits = F.sigmoid(logits)
            logits = torch.where(logits > 0.5, 1, 0)
            logits = multilabel_to_multiclass(
                labels=logits, value_for_other_class=value_for_other_class
            )  # Convert them to multiclass. E.g. [1, 0, 0, 0] -> 0

            y_true.extend(labels.tolist())
            y_pred.extend(logits.tolist())

    test_f1_macro = f1_score(
        y_true=y_true, y_pred=y_pred, average="macro"
    )  # use macro because our dataset is imbalanced.
    run.summary["test_f1"] = test_f1_macro
    logger.info(f"Test set f1: {test_f1_macro}")

    # NOTE: comment out the following line if you don't have an "Other" class.
    # label2id["Other"] = value_for_other_class

    report = classification_report(
        y_true,
        y_pred,
        labels=np.unique(y_true),
        target_names=list(label2id.keys()),
        zero_division=0,
    )
    logger.info(f"Classification Report:\n{report}")

    with open(
        os.path.join(model_path, "test_report.txt"),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write(report)

    fig_dir_path = os.path.join(model_path, "figs")
    os.makedirs(fig_dir_path, exist_ok=True)
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        classes=list(label2id.keys()),
        run=run,
        saveto=os.path.join(fig_dir_path, "cm.png"),
    )

    run.finish()
