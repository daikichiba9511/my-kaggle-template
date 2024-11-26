from collections.abc import Callable
from typing import Any, TypeAlias

import torch
import torch.nn as nn

LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_loss_fn(loss_name: str, loss_params: dict[str, Any]) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_params)
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**loss_params)
    if loss_name == "MSELoss":
        return nn.MSELoss(**loss_params)
    if loss_name == "L1Loss":
        return nn.L1Loss(**loss_params)
    raise ValueError(f"Unknown loss name: {loss_name}")
