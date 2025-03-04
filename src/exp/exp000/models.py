# pyright: reportPrivateImportUsage=false
#
# turn off lint for
# * timm.create_model
#
# Referenece:
# https://github.com/microsoft/pyright/blob/59dddc139c9929d3dc1ec8f9047d067e23404391/docs/configuration.md?plain=1#L158
from typing import Any, TypeAlias, cast

import timm
import torch
import torch.nn as nn
from timm.utils import model_ema


class CustomHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class CustomModel(nn.Module):
    """
    References:
    1. [数百種類のモデルを備える最強画像認識ライブラリ　「timm」のお手軽な使い方](https://logmi.jp/main/technology/325674)
    """
    def __init__(
        self,
        pretrained: bool = True,
        backbone: str = "convnext_tiny.in12k_ft_in1k",
        num_classes: int = 0,
        in_channels: int = 3,
        out_channels: int = 1,
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            num_classes=num_classes,
            features_only=features_only,
        )
        self.num_features = self.backbone.num_features
        self.head = CustomHead(self.num_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


Models: TypeAlias = CustomModel


def get_model(
    model_arch: str, model_params: dict[str, Any], use_ema: bool = True, ema_params: dict[str, Any] | None = None
) -> tuple[Models, model_ema.ModelEmaV3 | None]:
    if ema_params is None:
        ema_params = {"decal": 0.995, "update_after_step": 100, "use_warmup": True, "warmup_power": 0.75}

    if model_arch == "SimpleNN":
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
