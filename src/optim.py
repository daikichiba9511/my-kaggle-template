from typing import Any, TypeAlias

import torch
import transformers
from torch.optim import lr_scheduler


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
        optimizer = torch.optim.AdamW(model.parameters(), **optmizer_params)
        return optimizer
    raise ValueError(f"Unknown optimizer name: {optimizer_name}")


Schedulers: TypeAlias = lr_scheduler.LRScheduler | lr_scheduler.LambdaLR


def get_scheduler(
    scheduler_name: str, scheduler_params: dict[str, Any], optimizer: torch.optim.Optimizer
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
    raise ValueError(f"Unknown scheduler name: {scheduler_name}")
