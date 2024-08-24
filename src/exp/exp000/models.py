from typing import Any, TypeAlias, cast

import torch
import torch.nn as nn
from timm.utils import ModelEmaV3


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


Models: TypeAlias = Model


def get_model(model_name: str, model_params: dict[str, Any]) -> tuple[Models, ModelEmaV3]:
    if model_name == "Model":
        model = Model(**model_params)
        ema_model = ModelEmaV3(model, decay=0.9998)
        return model, ema_model
    raise ValueError(f"Unknown model name: {model_name}")


def compile_models(
    model: Models, ema_model: ModelEmaV3, compile_mode: str = "max-autotune", dynamic: bool = False
) -> tuple[Models, ModelEmaV3]:
    compiled_model = torch.compile(model, mode=compile_mode, dynamic=dynamic)
    compiled_model = cast(Models, compiled_model)
    compiled_ema_model = torch.compile(ema_model, mode=compile_mode, dynamic=dynamic)
    compiled_ema_model = cast(ModelEmaV3, compiled_ema_model)
    return compiled_model, compiled_ema_model


if __name__ == "__main__":
    model_name = "Model"
    model_params: dict[str, Any] = {}
    x = torch.randn(1, 3, 224, 224)

    model, ema_model = get_model(model_name, model_params=model_params)
    y = model(x)
    print(y)
    print(ema_model)
    print("Done!")
