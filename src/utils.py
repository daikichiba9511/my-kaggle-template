import contextlib
import functools
import json
import math
import multiprocessing as mp
import os
import pathlib
import pprint
import random
import subprocess
import time
from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from logging import getLogger
from typing import Any, Generator, Literal, TypeVar, overload

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import psutil
import seaborn as sns
import torch
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


def dbg(**kwargs: Any]) -> None:
    print("\n ********** DEBUG INFO ********* \n")
    print(kwargs)
    if kwargs.get("stop"):
        try:
            __import__("ipdb").set_trace()
        except ImportError:
            __import__("pdb").set_trace()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e from None


def get_model_param_size(model: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(model.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def show_model(model: torch.nn.Module, input_shape: tuple[int, ...] = (3, 224, 224)) -> None:
    summary(model, input_size=input_shape)


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


def to_heatmap(*, x: npt.NDArray[np.integer], y: npt.NDArray[np.integer]) -> npt.NDArray[np.int32]:
    """make heatmap array from x and y

    Args:
        x: array of x-axis.
        y: array of y-axis.

    Returns:
        heatmap: array of heatmap.
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    n_class_x = len(x_unique)
    n_class_y = len(y_unique)
    heatmap = np.zeros((n_class_x, n_class_y), dtype=np.int32)
    for i, class_x in enumerate(x_unique):
        for j, class_y in enumerate(y_unique):
            heatmap[i, j] = np.sum((x == class_x) & (y == class_y))
    return heatmap


def run_cmd(cmd: Sequence[str]) -> None:
    cmd_str = " ".join(cmd)
    print(f"Run command: {cmd_str}")
    subprocess.run(cmd, check=True)


_T_mp = TypeVar("_T_mp")
_S_mp = TypeVar("_S_mp")


@overload
def call_mp_unordered(
    fn: Callable[[_S_mp], _T_mp],
    containers: Collection[_S_mp],
    with_progress: Literal[True] = True,
    desc: str | None = None,
    n_jobs: int = -1,
) -> list[_T_mp]: ...


@overload
def call_mp_unordered(
    fn: Callable[[_S_mp], _T_mp],
    containers: Iterable[_S_mp],
    with_progress: Literal[False] = False,
    desc: str | None = None,
    n_jobs: int = -1,
) -> list[_T_mp]: ...


def call_mp_unordered(
    fn: Callable[[_S_mp], _T_mp],
    containers: Collection[_S_mp] | Iterable[_S_mp],
    with_progress: bool = False,
    desc: str | None = None,
    n_jobs: int = -1,
) -> list[_T_mp]:
    if desc is None:
        desc = "call func in multiprocessing"
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        if with_progress and isinstance(containers, Collection):
            return list(
                tqdm(
                    pool.imap_unordered(fn, containers),
                    total=len(containers),
                    desc=desc,
                    dynamic_ncols=True,
                )
            )
        return list(pool.imap_unordered(fn, containers))


def call_mp_ordered(
    fn: Callable[[_S_mp], _T_mp],
    containers: Sequence[_S_mp] | npt.NDArray,
    with_progress: bool = False,
    desc: str | None = None,
    n_jobs: int = -1,
) -> list[_T_mp]:
    if desc is None:
        desc = "call func in multiprocessing"
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        if with_progress:
            return list(
                tqdm(
                    pool.map(fn, containers),
                    total=len(containers),
                    desc=desc,
                    dynamic_ncols=True,
                )
            )
        return list(pool.map(fn, containers))


def calc_quantiles(x: npt.NDArray, n_splits: int) -> list[float]:
    return np.linspace(0, 1, n_splits + 1).tolist()


def _make_quantile_axis(x: npt.NDArray, quantiles: Sequence[float]) -> list[str]:
    axis_labels: list[str] = []
    for i in range(len(quantiles) - 1):
        start = x[i].item()
        if isinstance(start, float):
            start = round(start, 3)
        end = x[i + 1]
        if isinstance(end, float):
            end = round(end, 3)
        axis_labels.append(f"{start}-{end}")
    return axis_labels


def quantile_pivot_table(
    x: npt.NDArray, y: npt.NDArray, row_quantiles: Sequence[float], col_quantiles: Sequence[float]
) -> pl.DataFrame:
    """
    Create a pivot table based on n-th quantiles for rows and m-th quantiles for columns,
    using counts as the aggregation.
    """
    row_bins = np.quantile(x, row_quantiles).astype(x.dtype)
    col_bins = np.quantile(y, col_quantiles).astype(y.dtype)
    row_bin_indices = np.digitize(x, row_bins, right=True) - 1
    col_bin_indices = np.digitize(y, col_bins, right=True) - 1

    bins = np.vstack([row_bin_indices, col_bin_indices]).T
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    pivot_table = np.zeros((len(row_quantiles) - 1, len(col_quantiles) - 1), dtype=np.int32)
    for (row_bin, col_bin), count in zip(unique_bins, counts):
        if 0 <= row_bin < pivot_table.shape[0] and 0 <= col_bin < pivot_table.shape[1]:
            pivot_table[row_bin, col_bin] = count

    cols = _make_quantile_axis(col_bins, col_quantiles)
    rows = _make_quantile_axis(row_bins, row_quantiles)
    df = pl.DataFrame(pivot_table, schema=cols)
    df = df.with_columns(pl.Series("row_bin/col_bin", rows)).select(["row_bin/col_bin", *cols])
    return df


_T_pipe = TypeVar("_T_pipe")


def pipe(*func: Callable[[_T_pipe], _T_pipe]) -> Callable[[_T_pipe], _T_pipe]:
    @functools.wraps(pipe)
    def _inner(x: _T_pipe) -> _T_pipe:
        for fn in func:
            x = fn(x)
        return x

    return _inner


def save_as_json(obj: list | dict, save_fp: pathlib.Path, ensure_ascii: bool = False) -> None:
    with save_fp.open("w") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii)


def load_json(fp: pathlib.Path) -> object:
    with fp.open("r") as f:
        obj = json.load(f)
    return obj


# =============================================================================
# Test
# =============================================================================


def _test_quantile_pivot_table() -> None:
    from src import vis

    np.random.seed(42)
    x = np.random.randint(0, 100, 100)
    y = np.random.randint(0, 100, 100)
    # y = x**2 - 2 * x + 1 - np.random.randint(0, 100, 100)
    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    n_splits_row, n_splits_col = (5, 5)
    row_quantiles = calc_quantiles(x, n_splits_row)
    col_quantiles = calc_quantiles(y, n_splits_col)
    df = quantile_pivot_table(x, y, row_quantiles, col_quantiles)
    print(df)
    vis.plot_matrix(
        df.drop("row_bin/col_bin").to_numpy(),
        x_ticks_labels=df["row_bin/col_bin"].to_numpy().tolist(),
        y_ticks_labels=df.drop("row_bin/col_bin").columns,
        label_x="x",
        label_y="y",
        title="Quantile pivot table",
        save_fp=pathlib.Path("output/utils-test.png"),
    )


def _test_utils() -> None:
    _test_quantile_pivot_table()


if __name__ == "__main__":
    _test_utils()
