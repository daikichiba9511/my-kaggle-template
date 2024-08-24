import logging
import pathlib
import pprint
from typing import Sequence

import numpy as np
import numpy.typing as npt
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


def get_model_state_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    if hasattr(model, "_orig_mod"):
        # compileしたmodel
        logger.info("Detect compiled model. Accessing original model by _orig_mod")
        return model._orig_mod.state_dict()
    return model.state_dict()


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        is_maximise: bool = True,
        delta: float = 0.0,
    ) -> None:
        self._patience = patience
        self._is_maximise = is_maximise
        self.best_score = float("-inf") if self._is_maximise else float("inf")
        self._delta = delta
        self._reset_counter()
        self.is_improved = False

    def _can_update(self, score: float, best_score: float) -> bool:
        if self._is_maximise:
            self.is_improved = score + self._delta > best_score
            return self.is_improved
        else:
            self.is_improved = score + self._delta < best_score
            return self.is_improved

    def _reset_counter(self) -> None:
        self._counter = 0

    def _update_counter(self) -> None:
        self._counter += 1

    def _save(self, model: nn.Module, save_path: pathlib.Path) -> None:
        state = get_model_state_dict(model)
        torch.save(state, save_path)
        logger.info(f"Saved model {model.__class__.__name__}({type(model)}) to {save_path}")

    def check(self, score: float, model: nn.Module, save_path: pathlib.Path) -> None:
        if self._can_update(score, self.best_score):
            logger.info(f"Score improved from {self.best_score} to {score}")
            self.best_score = score
            self._reset_counter()
            self._save(model, save_path)
        else:
            self._update_counter()
            logger.info(
                f"EarlyStopping counter: {self._counter} out of {self._patience}. " + f"best: {self.best_score}"
            )

    @property
    def is_early_stopping(self) -> bool:
        return self._counter >= self._patience


class MetricsMonitor:
    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = metrics
        self._metrics_df = pl.DataFrame()

    def update(self, metrics: dict[str, float | int]) -> None:
        if "epoch" in metrics:
            raise ValueError("epoch is reserved word. Please use another key name")

        _metrics = pl.from_dict({k: [v] for k, v in metrics.items()})
        if self._metrics_df.is_empty():
            self._metrics_df = _metrics
        else:
            self._metrics_df = pl.concat([self._metrics_df, _metrics], how="vertical").sort(by="epoch")

        if wandb.run is not None:
            wandb.log(metrics)
        logger.info(f"Metrics updated: {pprint.pformat(metrics)}")

    def show(self, log_interval: int = 1) -> None:
        """print metrics to logger"""
        logging_metrics = self._metrics_df.filter(
            pl.col("epoch").is_in(list(range(0, len(self._metrics_df), log_interval)))
        )
        logger.info(f"\n{logging_metrics.to_pandas(use_pyarrow_extension_array=True).to_markdown()}")

    def plot(
        self,
        save_path: pathlib.Path,
        col: str | Sequence[str],
        figsize: tuple[int, int] = (14, 12),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        assert isinstance(ax, axes.Axes)
        assert isinstance(fig, figure.Figure)
        if isinstance(col, str):
            col = [col]
        for c in col:
            data = self._metrics_df[c].to_numpy()
            ax.plot(data, label=c)

        ax.set_xlabel("epoch", fontsize="small")
        ax.set_ylabel(",".join(col), fontsize="small")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close("all")

    def save(self, save_path: pathlib.Path, fold: int) -> None:
        self._metrics_df = self._metrics_df.with_columns(fold=pl.lit(fold))
        self._metrics_df.write_csv(save_path)
        logger.info(f"Saved metrics to {save_path}")


def make_oof(
    x: npt.NDArray,
    y: npt.NDArray,
    y_pred: npt.NDArray,
    x_names: list[str],
    y_names: list[str],
    sample_id: list[str] | None = None,
) -> pl.DataFrame:
    y_pred_names = [f"{y_name}_pred" for y_name in y_names]
    x_df = pl.DataFrame(x, x_names, orient="row")
    y_df = pl.DataFrame(y, y_names, orient="row")
    y_pred_df = pl.DataFrame(y_pred, y_pred_names, orient="row")
    if sample_id is None:
        return pl.concat([x_df, y_df, y_pred_df], how="horizontal")
    id_df = pl.DataFrame(sample_id, ["sample_id"], orient="row")
    return pl.concat([id_df, x_df, y_df, y_pred_df], how="horizontal")
