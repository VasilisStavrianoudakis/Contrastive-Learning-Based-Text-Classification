import itertools
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def init_logging(logger_name: str, log_level: str = logging.INFO):
    """
    Initializes and configures a logger with a specified name. The logger outputs
    INFO level logs to the console with a specific format including timestamp,
    log level, logger name, and the message.

    Args:
        logger_name (str): The name of the logger to initialize.
        log_level (str): The log level to set.

    Returns:
        logging.Logger: The configured logger instance.

    The logger is set to INFO level and outputs log messages to the console
    (via StreamHandler) in the format:
        [DD/MM/YYYY HH:MM:SS]: [LOG_LEVEL]: [LOGGER_NAME]: [MESSAGE]
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # StreamHandler to output logs to console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Log format including date, log level, logger name, and message
    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: %(name)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def open_json(filepath: str) -> Any:
    """Open a JSON file and return the data.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        Any: The data from the JSON file.
    """
    with open(filepath, "r", encoding="utf8") as fp:
        data = json.load(fp)
    return data


def write_json(data: Any, filepath: str) -> None:
    """Write data to a JSON file.

    Args:
        data (Any): The data to write to the JSON file.
        filepath (str): The path to the JSON file.
    """
    with open(filepath, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def get_colors(num_colors: int):
    """Get a list of colors.

    Args:
        num_colors (int): The number of colors to get.

    Returns:
        List[str]: A list of colors.
    """
    if num_colors < 5:
        colors = list(mcolors.BASE_COLORS)
        colors = [col for col in colors if "w" not in col and col not in ["r", "b"]]
    else:
        colors = list(mcolors.CSS4_COLORS)
        colors = [
            col
            for col in colors
            if "light" not in col and col not in ["white", "snow", "red", "blue"]
        ]
    colors = np.random.choice(colors, size=num_colors, replace=False)
    return colors


def plot_info(
    training_val,
    validation_val,
    y_name,
    saveto: str,
    extra_epochs: int = -1,
    events: Dict[str, int] = {},
):
    """Plot the training and validation curves.

    Args:
        training_val (List[float]): The training values.
        validation_val (List[float]): The validation values.
        y_name (str): The name of the y-axis.
        saveto (str): The path to save the plot.
        extra_epochs (int): The number of extra epochs.
        events (Dict[str, int]): The events to plot.
    """
    plt.figure()
    if events:
        colors = get_colors(len(events))
        for i, (event, x) in enumerate(events.items()):
            plt.axvline(x=x, color=colors[i], label=event, linestyle="--")

    epochs = (
        len(training_val) - extra_epochs if extra_epochs != -1 else len(training_val)
    )
    n_bins = epochs
    if extra_epochs != -1:
        if events["best_model"] + extra_epochs > epochs:
            n_bins = events["best_model"] + extra_epochs
        x_axis = list(range(events["best_model"] + 1))
        plt.plot(
            x_axis, training_val[: events["best_model"] + 1], "r-", label="training"
        )
        plt.plot(
            x_axis, validation_val[: events["best_model"] + 1], "b-", label="validation"
        )

        x_axis = list(
            range(events["best_model"], events["best_model"] + extra_epochs + 1)
        )
        extra_epochs_vals = training_val[epochs:]
        extra_epochs_vals.insert(0, training_val[events["best_model"]])
        plt.plot(x_axis, extra_epochs_vals, "r-")

        x_axis = list(range(events["best_model"], epochs))
        plt.plot(x_axis, training_val[events["best_model"] : epochs], "r--")
        plt.plot(x_axis, validation_val[events["best_model"] : epochs], "b--")
    else:
        x_axis = list(range(epochs))
        plt.plot(x_axis, training_val, "r-", label="training")
        x_axis = list(range(epochs))
        # plt.plot(x_axis, training_val, "r--", label="training")
        plt.plot(x_axis, validation_val, "b-", label="validation")

    if n_bins > 50:
        ticks = [num for num in range(n_bins) if num % 5 == 0]
    else:
        ticks = [num for num in range(n_bins) if num % 2 == 0]
    plt.xticks(ticks)
    plt.xticks(ticks, rotation=45, ha="right")

    plt.title(y_name)
    plt.ylabel(y_name)
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(saveto, bbox_inches="tight")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    run: Optional[Any] = None,
    saveto: Optional[str] = None,
):
    """Plot the confusion matrix.

    Args:
        y_true (List[int]): The true labels.
        y_pred (List[int]): The predicted labels.
        classes (List[str]): The classes.
        run (Optional[Any]): The run.
        saveto (Optional[str]): The path to save the plot.
    """
    # Calculate the cm.
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    classes = [cl.lower()[:20] for cl in classes]
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if run is not None:
        run.log({"confusion_matrix": plt})

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    # plt.show()


def get_examples_per_class(
    df: pd.DataFrame, label_column: str, diff_from_second: float = 2
) -> Dict[str, int]:
    """Get the examples per class.

    Args:
        df (pd.DataFrame): The dataframe.
        label_column (str): The label column.
        diff_from_second (float): The difference from the second class.

    Returns:
        Dict[str, int]: The examples per class.
    """
    cl_distribution = df[label_column].value_counts(dropna=False).to_dict()
    cl_distribution = {
        k: v
        for k, v in sorted(
            cl_distribution.items(), key=lambda item: item[1], reverse=True
        )
    }

    keys = list(cl_distribution.keys())
    vals = list(cl_distribution.values())
    if vals[0] > diff_from_second * vals[1]:
        cl_distribution[keys[0]] = int(diff_from_second * vals[1])
    return cl_distribution


class DummyLogger:

    def __init__(self, *args, **kwargs):
        """Initialize the dummy logger.

        Args:
            *args: The arguments.
            **kwargs: The keyword arguments.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return None

    def __setitem__(self, key, val):
        pass
