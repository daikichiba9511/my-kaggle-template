from collections.abc import Callable
from typing import Any, TypeAlias

import torch
import torch.nn as nn

LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]
"""(pred, target) -> {"loss": loss}"""


def wrap_loss_fn(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> LossFn:
    def wrapped_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = loss_fn(pred, target)
        return {"loss": loss}

    return wrapped_loss_fn


def get_loss_fn(loss_name: str, loss_params: dict[str, Any]) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        return wrap_loss_fn(nn.BCEWithLogitsLoss(**loss_params))
    if loss_name == "CrossEntropyLoss":
        return wrap_loss_fn(nn.CrossEntropyLoss(**loss_params))
    if loss_name == "MSELoss":
        return wrap_loss_fn(nn.MSELoss(**loss_params))
    if loss_name == "L1Loss":
        return wrap_loss_fn(nn.L1Loss(**loss_params))
    raise ValueError(f"Unknown loss name: {loss_name}")
