import contextlib
import math
import os
import pathlib
import pprint
import random
import subprocess
import time
from logging import getLogger
from typing import Any, Generator

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import psutil
import seaborn as sns
import torch
from matplotlib import axes, figure
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torchinfo import summary
from tqdm.auto import tqdm

logger = getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


def standarize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    xmin = x.min(dim).values
    xmax = x.max(dim).values
    return (x - xmin) / (xmax - xmin + eps)


def standarize_np(x: npt.NDArray, axis: int = 0, eps: float = 1e-8) -> npt.NDArray:
    xmin = x.min(axis)
    xmax = x.max(axis)
    return (x - xmin) / (xmax - xmin + eps)


@contextlib.contextmanager
def trace(title: str) -> Generator[None, None, None]:
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    duration = time.time() - t0
    duration_min = duration / 60
    msg = f"{title}: {m1:.2f}GB ({sign}{delta_mem:.2f}GB):{duration:.4f}s ({duration_min:3f}m)"
    print(f"\n{msg}\n")


@contextlib.contextmanager
def trace_with_cuda(title: str) -> Generator[None, None, None]:
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    allocated0 = torch.cuda.memory_allocated() / 10**9
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    duration = time.time() - t0
    duration_min = duration / 60

    allocated1 = torch.cuda.memory_allocated() / 10**9
    delta_alloc = allocated1 - allocated0
    sign_alloc = "+" if delta_alloc >= 0 else "-"

    msg = "\n".join([
        f"{title}: => RAM:{m1:.2f}GB({sign}{delta_mem:.2f}GB) "
        f"=> VRAM:{allocated1:.2f}GB({sign_alloc}{delta_alloc:.2f}) => DUR:{duration:.4f}s({duration_min:3f}m)"
    ])
    print(f"\n{msg}\n")


def get_commit_hash_head() -> str:
    """get commit hash"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=True)
    return result.stdout.decode("utf-8")[:-1]


def pinfo(msg: dict[str, Any]) -> None:
    logger.info(pprint.pformat(msg))


def reduce_memory_usage_pl(df: pl.DataFrame, name: str) -> pl.DataFrame:
    print(f"Memory usage of dataframe {name} is {round(df.estimated_size('mb'), 2)} MB")
    numeric_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    numeric_float_types = [pl.Float32, pl.Float64]
    float32_tiny = np.finfo(np.float32).tiny.astype(np.float64)
    float32_min = np.finfo(np.float32).min.astype(np.float64)
    float32_max = np.finfo(np.float32).max.astype(np.float64)
    for col in tqdm(df.columns, total=len(df.columns)):
        col_type = df[col].dtype
        c_min = df[col].to_numpy().min()
        c_max = df[col].to_numpy().max()
        if col_type in numeric_int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in numeric_float_types:
            if (
                (float32_min < c_min < float32_max)
                and (float32_min < c_max < float32_max)
                and (abs(c_min) > float32_tiny)  # 規格化定数で丸め込み誤差を補正
                and (abs(c_max) > float32_tiny)
            ):
                df = df.with_columns(df[col].cast(pl.Float32).alias(col))
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))
    print(f"Memory usage of dataframe {name} became {round(df.estimated_size('mb'), 2)} MB")
    return df


def save_as_pickle(obj: Any, save_fp: pathlib.Path) -> None:
    with save_fp.open("wb") as f:
        joblib.dump(obj, f)


def load_pickle(fp: pathlib.Path) -> Any:
    with fp.open("rb") as f:
        obj = joblib.load(f)
    return obj


def dbg(**kwargs: dict[Any, Any]) -> None:
    print("\n ********** DEBUG INFO ********* \n")
    print(kwargs)
    if kwargs.get("stop"):
        __import__("pdb").set_trace()


def get_model_param_size(model: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(model.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def show_model(model: torch.nn.Module, input_shape: tuple[int, ...] = (3, 224, 224)) -> None:
    summary(model, input_size=input_shape)


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
            title = "Confusion matrix, without normalization"

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
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel("Predicted label", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
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


def save_importances(importances: pl.DataFrame, save_fp: pathlib.Path, figsize: tuple[int, int] = (16, 30)) -> None:
    mean_gain = importances[["gain", "feature"]].group_by("feature").mean().rename({"gain": "mean_gain"})
    importances = importances.join(mean_gain, on="feature")
    plt.figure(figsize=figsize)
    sns.barplot(
        x="mean_gain",
        y="feature",
        data=importances.sort("mean_gain", descending=True)[:300].to_pandas(),
        color="skyblue",
    )
    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close("all")
