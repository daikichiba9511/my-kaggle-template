import logging
import pathlib
import pprint
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy.typing as npt
import polars as pl
import torch
import torch.nn as nn
import wandb
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from timm.utils import model_ema
from torch.amp import grad_scaler
from typing_extensions import NotRequired

from src import optim, utils

logger = logging.getLogger(__name__)


def freeze(model: nn.Module, keys: Sequence[str]) -> None:
    """Freeze model parameters with keys"""
    for name, param in model.named_parameters():
        for key in keys:
            if key in name:
                param.requires_grad = False


def unfreeze(model: nn.Module, keys: Sequence[str]) -> None:
    """Unfreeze model parameters with keys"""
    for name, param in model.named_parameters():
        for key in keys:
            if key in name:
                param.requires_grad = True


class AvgMeterDict(TypedDict):
    name: str
    avg: float
    raw_values: list[float | int]


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

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.raws: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        if utils.is_inf(value) or utils.is_nan(value):
            logger.info("Skip nan or inf value")
            return None

        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.raws.append(value)

    def to_dict(self) -> AvgMeterDict:
        return {
            "name": self.name,
            "avg": self.avg,
            "raw_values": self.raws,
        }


class AverageMeters:
    def __init__(self, names: Sequence[str]) -> None:
        self.meters = {name: AverageMeter(name) for name in names}

    def __str__(self) -> str:
        return "\n".join([str(meter) for meter in self.meters.values()])

    def __repr__(self) -> str:
        return self.__str__()

    def update(self, values: dict[str, float | int], n: int = 1) -> None:
        for name, value in values.items():
            if name not in self.meters:
                raise ValueError(f"Name {name} is not in the meters" + f"Available names are {self.meters.keys()}")
            self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def to_dict(self) -> dict[str, AvgMeterDict]:
        return {name: meter.to_dict() for name, meter in self.meters.items()}


def get_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Get model state dict

    Args:
        model : nn.Module or compiled model

    Returns:
        state_dict : model state dict
    """
    if hasattr(model, "_orig_mod"):
        # compileしたmodel
        logger.info("Detect compiled model. Accessing original model by _orig_mod")
        return model._orig_mod.state_dict()
    return model.state_dict()


class FullCKPT(TypedDict):
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    epoch: int
    ema_model: NotRequired[dict[str, Any]]


class WeightOnlyCKPT(TypedDict):
    model: dict[str, Any]


CKPT = FullCKPT | WeightOnlyCKPT


def make_checkpoints(
    model: nn.Module,
    optimizer: torch.optim.optimizer.Optimizer,
    scheduler: optim.Schedulers,
    scaler: grad_scaler.GradScaler,
    epoch: int,
    ema_model: model_ema.ModelEmaV3 | None = None,
) -> FullCKPT:
    ckpts: FullCKPT = {
        "model": get_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    if ema_model is not None:
        ckpts["ema_model"] = ema_model.state_dict()
    return ckpts


def load_checkpoints(
    model: nn.Module,
    optimizer: torch.optim.optimizer.Optimizer,
    scheduler: optim.Schedulers,
    scaler: grad_scaler.GradScaler,
    checkpoint: FullCKPT,
    ema_model: model_ema.ModelEmaV3 | None = None,
) -> None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if ema_model is not None and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
    elif ema_model is not None and "ema_model" not in checkpoint:
        raise ValueError("ema_model is not in the checkpoint. Please check the checkpoint keys : ", checkpoint.keys())


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
    def __init__(self, show_metrics: Sequence[str]) -> None:
        self.show_metrics = show_metrics
        self._metrics_df = pl.DataFrame()

    def update(self, metrics: dict[str, float | int]) -> None:
        _metrics = pl.from_dict({k: [v] for k, v in metrics.items()})
        if self._metrics_df.is_empty():
            self._metrics_df = _metrics
        else:
            self._metrics_df = pl.concat([self._metrics_df, _metrics], how="vertical").sort(by="epoch")

        if wandb.run is not None:
            wandb.log(metrics, step=int(metrics["epoch"]))

        logger.info(f"Metrics updated:\n {pprint.pformat(metrics)}")

    def show(self, use_logger: bool = False, log_interval: int = 1) -> None:
        """print metrics to logger"""
        logging_metrics = self._metrics_df.select(self.show_metrics).filter(
            pl.col("epoch").is_in(list(range(0, len(self._metrics_df), log_interval)))
        )
        msg = f"\n{logging_metrics.to_pandas().to_markdown()}\n"
        logger.info(msg) if use_logger else print(msg)

    def plot(
        self,
        save_path: pathlib.Path,
        col: str | Sequence[str],
        figsize: tuple[int, int] = (14, 12),
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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

    def save(self, save_path: pathlib.Path, fold: int | None = None) -> None:
        if fold is not None:
            self._metrics_df = self._metrics_df.with_columns(fold=pl.lit(fold))
        self._metrics_df.write_csv(save_path)
        logger.info(f"Saved metrics to {save_path}")


def step(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.optimizer.Optimizer,
    loss: torch.Tensor,
    max_norm: float | None = None,
    scaler: grad_scaler.GradScaler | None = None,
    ema_model: model_ema.ModelEmaV3 | None = None,
    scheduler: optim.Schedulers | None = None,
    grad_accum_steps: int = 1,
    num_updates: int | None = None,
    skip_nan: bool = False,
    raise_nan: bool = False,
) -> torch.Tensor:
    """Step for training

    Args:
        step : step number
        model : nn.Module
        optimizer : torch.optim.optimizer.Optimizer
        loss : loss tensor
        max_norm : If not None, clip grad norm. Default is None. If None, not clip grad norm
        scaler : If not None, use scaler.step() and scaler.update()
        ema_model : If not None, ema_model update module attribute of ema_model using model
        scheduler : If not None, scheduler.step() is called
        grad_accum_steps : If not None, accumulate grad for grad_accum_steps times
        num_updates : If not None, ema_model update with num_updates
        skip_nan : If True, skip nan loss
        raise_nan : If True, raise ValueError when loss has nan value

    Returns:
        norm: If max_norm is not None, return grad norm of model parameters
    """
    has_nan = torch.isnan(loss).any()
    if has_nan and skip_nan:
        logger.info("Skip nan loss")
        return torch.tensor(0.0)

    if has_nan and raise_nan:
        raise ValueError("Loss has nan value")

    norm = torch.tensor(0.0)
    if scaler is not None:
        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            if max_norm is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                ema_model.update(model, step=num_updates)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    else:
        loss.backward()

        if step % grad_accum_steps == 0:
            if max_norm is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()

            if ema_model is not None:
                ema_model.update(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return norm


def make_oof(
    x: npt.NDArray,
    y: npt.NDArray,
    y_pred: npt.NDArray,
    x_names: Sequence[str],
    y_names: Sequence[str],
    sample_id: Sequence[str] | None = None,
) -> pl.DataFrame:
    y_pred_names = [f"{y_name}_pred" for y_name in y_names]
    x_df = pl.DataFrame(x, x_names, orient="row")
    y_df = pl.DataFrame(y, y_names, orient="row")
    y_pred_df = pl.DataFrame(y_pred, y_pred_names, orient="row")
    if sample_id is None:
        return pl.concat([x_df, y_df, y_pred_df], how="horizontal")
    id_df = pl.DataFrame(sample_id, ["sample_id"], orient="row")
    return pl.concat([id_df, x_df, y_df, y_pred_df], how="horizontal")


class UpdateManager:
    def __init__(self, is_maximize: bool, n_epochs: int) -> None:
        self._is_maximize = is_maximize
        self._n_epochs = n_epochs
        self._best_score = float("-inf") if self._is_maximize else float("inf")

    def check_score(self, score: float) -> bool:
        if self._is_maximize:
            if score > self._best_score:
                self._best_score = score
                return True
        else:
            if score < self._best_score:
                self._best_score = score
                return True
        return False

    def check_epoch(self, epoch: int) -> bool:
        return epoch >= self._n_epochs

    @property
    def best_score(self) -> float:
        return self._best_score
