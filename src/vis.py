import pathlib
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from matplotlib import axes, cm, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from src import utils

_T_NBit = TypeVar("_T_NBit", bound=npt.NBitBase)


def bar(
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    x_label: str,
    y_label: str,
    title: str,
    ax: axes.Axes | None = None,
    label_type: Literal["center", "edge"] = "edge",
    fig_size: tuple[int, int] = (15, 10),
    save_fp: pathlib.Path | None = None,
) -> None:
    """plot bar"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = None
    if ax is None:
        raise ValueError("ax is None")
    bars = ax.bar(x, y, alpha=0.7, label=y_label)
    ax.bar_label(bars, label_type=label_type)
    ax.set_title(title)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(str, x)))
    ax.legend()
    if fig is not None:
        fig.tight_layout()
        fig.show()
    if save_fp is not None and fig is not None:
        fig.savefig(save_fp)


def _complement_lack_classes(
    y: npt.NDArray[np.int32], y_cnt: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    for i in range(4):
        if i not in y:
            y = np.append(y, i)
            y_cnt = np.append(y_cnt, 0)
    idx = np.argsort(y)
    y = y[idx]
    y_cnt = y_cnt[idx]
    return y, y_cnt


def dists(
    y_true: npt.NDArray[np.int32],
    y_pred: npt.NDArray[np.int32],
    save_fp: pathlib.Path,
    figsize: tuple[int, int] = (30, 20),
) -> None:
    # -- 予測値 & 正解値の分布
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    y_true_values, y_true_counts = np.unique(y_true, return_counts=True)
    y_true_values, y_true_counts = _complement_lack_classes(y_true_values, y_true_counts)  # type: ignore
    bar(x=y_true_values, y=y_true_counts, x_label="target", y_label="count", title="y_true", ax=ax[0, 0])
    y_pred_values, y_pred_counts = np.unique(y_pred, return_counts=True)
    y_pred_values, y_pred_counts = _complement_lack_classes(y_pred_values, y_pred_counts)  # type: ignore
    bar(x=y_pred_values, y=y_pred_counts, x_label="target", y_label="count", title="y_pred", ax=ax[0, 1])

    ax02 = ax[0, 2]
    bars = ax02.bar(y_true_values, y_true_counts, alpha=0.5, label="y_true")
    ax02.bar_label(bars, label_type="center")
    bars = ax02.bar(y_pred_values, y_pred_counts, alpha=0.5, label="y_pred")
    ax02.bar_label(bars)
    ax02.set_title("y_true vs y_pred")
    ax02.set_xticks(y_true_values)
    ax02.set(xlabel="target", ylabel="count", xticklabels=list(map(str, y_true_values)))
    ax02.legend()

    # -- 予測値 & 正解値のうち間違えた部分の分布
    miss = y_true != y_pred
    miss_y_true = y_true[miss]
    miss_y_pred = y_pred[miss]

    miss_y_true_values, miss_y_true_counts = np.unique(miss_y_true, return_counts=True)
    miss_y_true_values, miss_y_true_counts = _complement_lack_classes(miss_y_true_values, miss_y_true_counts)
    bar(
        x=miss_y_true_values, y=miss_y_true_counts, x_label="target", y_label="count", title="miss_y_true", ax=ax[1, 0]
    )
    miss_y_pred_values, miss_y_pred_counts = np.unique(miss_y_pred, return_counts=True)
    miss_y_pred_values, miss_y_pred_counts = _complement_lack_classes(miss_y_pred_values, miss_y_pred_counts)
    bar(
        x=miss_y_pred_values, y=miss_y_pred_counts, x_label="target", y_label="count", title="miss_y_pred", ax=ax[1, 1]
    )

    ax12 = ax[1, 2]
    bars = ax12.bar(miss_y_true_values, miss_y_true_counts, alpha=0.5, label="miss_y_true")
    ax12.bar_label(bars, label_type="center")
    bars = ax12.bar(miss_y_pred_values, miss_y_pred_counts, alpha=0.5, label="miss_y_pred")
    ax12.bar_label(bars)
    ax12.set_title("miss_y_true vs miss_y_pred")
    ax12.set_xticks(miss_y_pred_values)
    ax12.set(xlabel="target", ylabel="count", xticklabels=list(map(str, miss_y_true_values)))
    ax12.legend()

    fig.suptitle("Distribution of y_true and y_pred")
    fig.tight_layout()
    fig.savefig(save_fp)
    plt.close("all")


def heatmap(
    *,
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    x_label: str,
    y_label: str,
    save_fp: pathlib.Path | None = None,
) -> None:
    """plot heatmap

    Args:
        x: array of x-axis
        y: array of y-axis
        x_label: x-axis label
        y_label: y-axis label

    Returns:
        None
    """
    if not issubclass(x.dtype.type, np.integer):
        class_map_x = {class_: i for i, class_ in enumerate(np.unique(x))}
    else:
        class_map_x = None

    if not issubclass(y.dtype.type, np.integer):
        class_map_y = {class_: i for i, class_ in enumerate(np.unique(y))}
    else:
        class_map_y = None

    print(f"class_map_x: {class_map_x}, class_map_y: {class_map_y}")

    heatmap = utils.to_heatmap(x=x, y=y)
    max_size = np.max(heatmap.shape)
    fig, ax = plt.subplots(1, 1, figsize=(2 * max_size, 2 * max_size))
    sns.heatmap(heatmap, annot=True, ax=ax, cmap="Blues", fmt=".0f")
    ax.set(xlabel=x_label, ylabel=y_label, title=f"{x_label} vs {y_label}")
    if class_map_x is not None:
        ax.set_xticklabels(list(map(str, class_map_x.keys())))
    if class_map_y is not None:
        ax.set_yticklabels(list(map(str, class_map_y.keys())))
    fig.tight_layout()
    fig.show()
    if save_fp is not None:
        fig.savefig(save_fp)


def scatter(
    *,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    x_label: str,
    y_label: str,
    fig_size: tuple[int, int] = (15, 10),
    save_fp: pathlib.Path | None = None,
    ax: axes.Axes | None = None,
) -> None:
    """plot scatter

    Args:
        x: array of x-axis
        y: array of y-axis
        x_label: x-axis label
        y_label: y-axis label
        fig_size: figure size. [Width, Height]
        save_fp: save file path
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = None
    if ax is None:
        raise ValueError("ax is None")
    ax.scatter(x, y, alpha=0.7, marker="x", linewidths=1, label=f"{x_label} vs {y_label}")
    ax.set(xlabel=x_label, ylabel=y_label, title=f"{x_label} vs {y_label}")
    if fig is not None:
        fig.tight_layout()
        fig.show()
    if save_fp is not None and fig is not None:
        fig.savefig(save_fp)


def plot_images(
    images: torch.Tensor | npt.NDArray,
    title: str,
    save_path: pathlib.Path | None = None,
    figsize: tuple[int, int] = (30, 15),
) -> None:
    """
    Args:
        images: list of images to plot. (n, h, w, c) or (n, c, h, w)
    """
    if isinstance(images, torch.Tensor) and images.shape[-1] != 3:
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    elif isinstance(images, np.ndarray) and images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))

    n_rows = len(images)
    if n_rows > 5:
        raise ValueError("Too many images to plot")

    fig, ax = plt.subplots(1, n_rows, figsize=figsize)
    for i, img in enumerate(images):
        ax[i].imshow(img, label=f"image_{i}")
        ax[i].set_title(f"image_{i}", fontsize="small")

    # -- draw object overlay here

    # -- draw title & plot/save
    fig.suptitle(title, fontsize="small")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(str(save_path))
        plt.close("all")


def plot_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: npt.NDArray,
    normalize: bool = False,
    title: str | None = None,
    cmap: cm.Blues = cm.Blues,  # type: ignore
) -> tuple[figure.Figure, axes.Axes]:
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, w/o normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    assert isinstance(ax, axes.Axes)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, fontsize=25)
    ax.set_yticks(tick_marks, fontsize=25)
    ax.set_xlabel("Predicted label", fontsize=25)
    ax.set_ylabel("True label", fontsize=25)
    ax.set_title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    _figure = ax.figure
    assert isinstance(_figure, figure.Figure)
    cbar = _figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                fontsize=20,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig, ax
