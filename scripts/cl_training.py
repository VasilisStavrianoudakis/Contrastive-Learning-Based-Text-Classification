import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from imblearn.under_sampling import RandomUnderSampler
from losses import SCL
from torch.utils.data import DataLoader
from training_utils import DummyLogger, get_examples_per_class, init_logging, open_json
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

logger = init_logging(logger_name="cl_training", log_level=logging.DEBUG)


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = SCL()

    def compute_loss(self, model, inputs, return_outputs=False):
        # By default Trainer will place the labels inside inputs under the "labels" key.
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.pooler_output
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # Main evaluation loop
        val_losses = []
        self.model.eval()
        # We need to return the predictions.
        # But we don't calculate any prediction here so we will return the labels.
        all_preds = []
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                labels = inputs.pop("labels")
                outputs = self.model(**inputs)

                logits = outputs.pooler_output
                loss = self.loss_func(logits, labels)

                val_losses.append(loss.item())
                all_preds.append(labels.cpu())

        all_preds = np.concatenate(all_preds)
        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=None,
            metrics={"eval_loss": torch.tensor(val_losses).mean().item()},
            num_samples=all_preds.shape[0],
        )


def process_dataset(
    data,
    tokenizer: AutoTokenizer,
    text_column: str,
):
    data = tokenizer(
        data[text_column], truncation=True, padding="max_length", max_length=512
    )
    return data


def prune_layers(model: AutoModel, layers_to_keep: List[str]) -> AutoModel:
    """
    Remove (prune) layers from the model.

    Args:
        `layers_to_keep` (List[str]):
            The layers of the model we want to keep.
    """
    # remove transformer blocks
    new_layers = torch.nn.ModuleList(
        [
            layer_module
            for i, layer_module in enumerate(model.encoder.layer)
            if i in layers_to_keep
        ]
    )
    model.encoder.layer = new_layers
    model.config.num_hidden_layers = len(layers_to_keep)
    logger.debug(model)
    return model


def freeze_or_unfreeze_layers(
    model: AutoModel, layers: List[str], freeze: bool = True
) -> AutoModel:
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
    # File paths and variables.
    this_file_path = Path(__file__).resolve().parent
    data_dir_path = os.path.join(this_file_path, "..", "data")
    splits_dir_path = os.path.join(data_dir_path, "splits")
    inference_path = os.path.join(this_file_path, "..", "inference")

    config = open_json(filepath=os.path.join(this_file_path, "cl_config.json"))

    model_id = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    base_model_path = config["base_model_path"]
    contrastive_model_path = os.path.join(
        inference_path,
        "contrastive_learning",
        config["task"],
        f"{model_id}",
    )

    label_column = "labels"
    text_column = "processed_text"

    # Get the data.
    training_df: pd.DataFrame = pd.read_pickle(
        os.path.join(splits_dir_path, "train.pkl")
    )
    validation_df: pd.DataFrame = pd.read_pickle(
        os.path.join(splits_dir_path, "cl_validation.pkl")
    )

    # NOTE: comment out the following two lines if you don't have an "Other" class.
    # training_df["label"].fillna(training_df["label"].max() + 1, inplace=True)
    # validation_df["label"].fillna(validation_df["label"].max() + 1, inplace=True)

    training_df = training_df.rename(
        columns={
            "label": label_column,
        }
    ).reset_index(drop=True)
    validation_df = validation_df.rename(columns={"label": label_column}).reset_index(
        drop=True
    )

    logger.debug(
        f"Training set Distribution:\n{training_df[label_column].value_counts(dropna=False)}"
    )
    logger.debug(
        f"Validation set Distribution:\n{validation_df[label_column].value_counts(dropna=False)}"
    )

    if config["use_undersampler"]:
        training_ratio = get_examples_per_class(
            df=training_df,
            label_column=label_column,
            diff_from_second=config["diff_from_second_class"],
        )
        rus = RandomUnderSampler(random_state=0, sampling_strategy=training_ratio)
        training_df, _ = rus.fit_resample(training_df, training_df[label_column])

        training_df = training_df.reset_index(drop=True)

    config["training_examples"] = training_df.shape[0]
    config["validation_examples"] = validation_df.shape[0]
    # We need to drop the following columns from our datasets objects.
    remove_columns = training_df.columns.tolist()
    remove_columns.remove(label_column)
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
            job_type="Contrastive Learning",
        )
    else:
        run = DummyLogger()

    logger.info(
        f"Training set Distribution:\n{training_df[label_column].value_counts(dropna=False)}",
    )
    logger.info(
        f"Validation set Distribution:\n{validation_df[label_column].value_counts(dropna=False)}",
    )

    training_dataset = Dataset.from_pandas(df=training_df)
    validation_dataset = Dataset.from_pandas(df=validation_df)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    training_dataset = training_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column": text_column,
        },
        remove_columns=remove_columns,
    )
    validation_dataset = validation_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column": text_column,
        },
        remove_columns=remove_columns,
    )

    # Load the model.
    model = AutoModel.from_pretrained(
        base_model_path,
    )
    # Prune model layers.
    if config["keep_layers"] is not None:
        model = prune_layers(model=model, layers_to_keep=config["keep_layers"])

    model = freeze_or_unfreeze_layers(model=model, layers=["embeddings"], freeze=True)
    logger.debug(model)

    # NOTE: about load_best_model_at_end=False.
    # It is set to False because in contrastive learning we don't really care about model selection.
    # We want to save the model at the last epoch, regardless of its performance on the val set, because training the model longer results in better performance on the downstream task.
    training_args = TrainingArguments(
        output_dir=contrastive_model_path,
        overwrite_output_dir=True,
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="loss",
        load_best_model_at_end=False,  # NOTE: Read the comments above!
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        greater_is_better=False,
        save_total_limit=2,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to="wandb" if config["enable_wandb_logging"] else "none",
        remove_unused_columns=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()

    # Save the model and its tokenizer to load them together.
    trainer.save_model(contrastive_model_path)
    tokenizer.save_pretrained(contrastive_model_path)

    run.finish()
