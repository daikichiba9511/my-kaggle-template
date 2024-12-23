from typing import Any, TypeAlias, cast

import torch
import torch.nn as nn
from timm.utils import model_ema


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


Models: TypeAlias = CustomModel


def get_model(
    model_name: str, model_params: dict[str, Any], use_ema: bool = True, ema_params: dict[str, Any] | None = None
) -> tuple[Models, model_ema.ModelEmaV3 | None]:
    if ema_params is None:
        ema_params = {"decal": 0.995, "update_after_step": 100, "use_warmup": True, "warmup_power": 0.75}

    if model_name == "SimpleNN":
        model = CustomModel(**model_params)
        ema_model = model_ema.ModelEmaV3(model, **ema_params) if use_ema else None
        return model, ema_model
    raise ValueError(f"Unknown model name: {model_name}")


def compile_models(
    model: Models, ema_model: model_ema.ModelEmaV3 | None, compile_mode: str = "max-autotune", dynamic: bool = False
) -> tuple[Models, model_ema.ModelEmaV3 | None]:
    compiled_model = torch.compile(model, mode=compile_mode, dynamic=dynamic)
    compiled_model = cast(Models, compiled_model)
    if ema_model is None:
        return compiled_model, None
    compiled_ema_model = torch.compile(ema_model, mode=compile_mode, dynamic=dynamic)
    compiled_ema_model = cast(model_ema.ModelEmaV3, compiled_ema_model)
    return compiled_model, compiled_ema_model


if __name__ == "__main__":
    from torchinfo import summary

    model_name = "SimpleNN"
    model_params: dict[str, Any] = {}
    shape = (1, 3, 224, 224)
    x = torch.randn(*shape)

    model, ema_model = get_model(model_name, model_params=model_params)
    y = model(x)
    print(f"{y.shape=}")
    summary(model, input_size=shape)
    print("Done!")
