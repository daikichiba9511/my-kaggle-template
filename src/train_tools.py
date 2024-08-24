import logging
import pathlib
from typing import Literal, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import wandb
from matplotlib import axes, figure
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def freeze(model: nn.Module, keys: Sequence[str]) -> None:
    """Freeze model parameters with keys"""
    for name, param in model.named_parameters():
        for key in keys:
            if name in key:
                param.requires_grad = False


def unfreeze(model: nn.Module, keys: Sequence[str]) -> None:
    """Unfreeze model parameters with keys"""
    for name, param in model.named_parameters():
        for key in keys:
            if name in key:
                param.requires_grad = True


class AverageMeter:
    """Computes and stores the average and current value"""

    val: float
    avg: float
    sum: float | int
    count: int
    rows: list[float | int]

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.raws: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        if value in [np.nan, np.inf, -np.inf, float("inf"), float("-inf"), float("nan")]:
            logger.info("Skip nan or inf value")
            return None
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.raws.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {
            "name": self.name,
            "avg": self.avg,
            "raw_values": self.raws,
        }


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        direction: Literal["max", "min"] = "min",
        delta: float = 0.0,
    ) -> None:
        if direction not in ["max", "min"]:
            raise ValueError(f"{direction = }")

        self._patience = patience
        self._direction = direction
        self._counter = 0
        self.best_score = float("-inf") if direction == "max" else float("inf")
        self._delta = delta
        self.is_improved = False

    def _is_improved(self, score: float, best_score: float) -> bool:
        if self._direction == "max":
            self.is_improved = score + self._delta > best_score
            return self.is_improved
        else:
            self.is_improved = score + self._delta < best_score
            return self.is_improved

    def _save(self, model: nn.Module, save_path: pathlib.Path) -> None:
        if hasattr(model, "_orig_mod"):
            # compileしたmodel
            logger.info("Detect compiled model. Accessing original model by _orig_mod")
            state = model._orig_mod.state_dict()
        else:
            state = model.state_dict()
        torch.save(state, save_path)
        logger.info(f"Saved model {model.__class__.__name__}({type(model)}) to {save_path}")

    def check(self, score: float, model: nn.Module, save_path: pathlib.Path) -> None:
        if self._is_improved(score, self.best_score):
            logger.info(f"Score improved from {self.best_score} to {score}")
            self.best_score = score
            self._counter = 0
            self._save(model, save_path)
        else:
            self._counter += 1
            logger.info(
                f"EarlyStopping counter: {self._counter} out of {self._patience}. " + f"best: {self.best_score}"
            )

    @property
    def is_early_stop(self) -> bool:
        return self._counter >= self._patience


class MetricsMonitor:
    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = metrics
        self._metrics_df = pd.DataFrame(columns=[*metrics])  # type: ignore

    def update(self, metrics: dict[str, float | int]) -> None:
        epoch = cast(int, metrics.pop("epoch"))
        _metrics = pd.DataFrame(metrics, index=[epoch])  # type: ignore
        if self._metrics_df.empty:
            self._metrics_df = _metrics
        else:
            self._metrics_df = pd.concat([self._metrics_df, _metrics], axis=0)

        if wandb.run is not None:
            wandb.log(metrics)

    def show(self, log_interval: int = 1) -> None:
        """print metrics to logger"""
        logging_metrics: pd.DataFrame = self._metrics_df.iloc[list(range(0, len(self._metrics_df), log_interval))]
        logger.info(f"\n{logging_metrics.to_markdown()}")

    def plot(
        self,
        save_path: pathlib.Path,
        col: str | Sequence[str],
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        assert isinstance(ax, axes.Axes)
        assert isinstance(fig, figure.Figure)
        if isinstance(col, str):
            col = [col]
        for c in col:
            data = self._metrics_df[c].to_numpy()
            ax.plot(data, label=c)
        ax.set_xlabel("epoch")
        ax.set_ylabel(",".join(col))
        ax.legend()
        fig.savefig(save_path)

    def save(self, save_path: pathlib.Path, fold: int) -> None:
        self._metrics_df["fold"] = fold
        self._metrics_df.to_csv(save_path, index=False)


def make_oof(
    x: npt.NDArray,
    y: npt.NDArray,
    y_pred: npt.NDArray,
    feature_names: list[str],
    y_names: list[str],
    id: list[str] | None = None,
) -> pl.DataFrame:
    y_pred_names = [f"{y_name}_pred" for y_name in y_names]
    x_df = pl.DataFrame(x, feature_names, orient="row")
    y_df = pl.DataFrame(y, y_names, orient="row")
    y_pred_df = pl.DataFrame(y_pred, y_pred_names, orient="row")
    if id is None:
        return pl.concat([x_df, y_df, y_pred_df], how="horizontal")
    id_df = pl.DataFrame(id, ["sample_id"], orient="row")
    return pl.concat([id_df, x_df, y_df, y_pred_df], how="horizontal")
