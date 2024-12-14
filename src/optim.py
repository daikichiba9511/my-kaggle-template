from logging import getLogger
from typing import Any, Literal, TypeAlias, overload

import torch
import transformers
from torch.optim import lr_scheduler

logger = getLogger(__name__)


def get_params_no_decay(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay: tuple[str, ...] = ("bias", "ln.bias", "ln.weight"),
) -> list[dict[str, Any]]:
    model_params = list(model.named_parameters())
    print([k for k, _ in model_params])
    params = [
        {
            "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return params


def get_optimizer(
    optimizer_name: str, optmizer_params: dict[str, Any], model: torch.nn.Module
) -> torch.optim.Optimizer:
    """Get optimizer from optimizer name

    Args:
        optimizer_name (str): optimizer name
        optmizer_params (dict[str, Any]): optimizer parameters
        model (torch.nn.Module): model

    Returns:
        torch.optim.Optimizer: optimizer
    """
    if optimizer_name == "AdamW":
        model_params = get_params_no_decay(model, weight_decay=optmizer_params["weight_decay"])
        optimizer = torch.optim.AdamW(model_params, **optmizer_params)
        return optimizer
    raise ValueError(f"Unknown optimizer name: {optimizer_name}")


Schedulers: TypeAlias = lr_scheduler.LRScheduler | lr_scheduler.LambdaLR | lr_scheduler.CosineAnnealingWarmRestarts


def setup_scheduler_params(
    scheduler_params: dict[str, Any], num_step_per_epoch: int, n_epoch: int, warmup_epochs: int = 1
) -> dict[str, Any]:
    """Set up scheduler parameters

    Args:
        scheduler_params (dict[str, Any]): scheduler parameters
        num_step_per_epoch (int): number of step per epoch
        n_epoch (int): number of epoch
        warmup_epochs (int, optional): number of warmup epochs. Defaults to 1.
    """
    _scheduler_params = {**scheduler_params}
    num_total_steps = num_step_per_epoch * n_epoch
    _scheduler_params["num_training_steps"] = num_total_steps
    _scheduler_params["num_warmup_steps"] = int((num_total_steps / n_epoch) * warmup_epochs)
    logger.info(f"Update scheduler_params: {_scheduler_params}")
    return _scheduler_params


SchedulersLiteral: TypeAlias = Literal["CosineLRScheduler", "CosineAnnealingWarmRestarts"]


@overload
def get_scheduler(
    scheduler_name: Literal["CosineLRScheduler"], scheduler_params: dict[str, Any], optimizer: torch.optim.Optimizer
) -> lr_scheduler.LambdaLR: ...


@overload
def get_scheduler(
    scheduler_name: Literal["CosineAnnealingWarmRestarts"],
    scheduler_params: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> lr_scheduler.CosineAnnealingWarmRestarts: ...


def get_scheduler(
    scheduler_name: SchedulersLiteral,
    scheduler_params: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> Schedulers:
    """Get scheduler from scheduler name

    Args:
        scheduler_name (str): scheduler name
        scheduler_params (dict[str, Any]): scheduler parameters
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        Schedulers: scheduler.
    """
    if scheduler_name == "CosineLRScheduler":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, **scheduler_params)
        return scheduler
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
        return scheduler
    raise ValueError(f"Unknown scheduler name: {scheduler_name}")
