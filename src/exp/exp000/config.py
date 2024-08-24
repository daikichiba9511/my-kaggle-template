import pathlib

import pydantic
import torch

from src import constants

EXP_NO = __file__.split("/")[-2]
DESCRIPTION = """
simple baseline
"""


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str = pydantic.Field(default=EXP_NO)
    description: str = pydantic.Field(default=DESCRIPTION)

    # -- General
    is_debug: bool = pydantic.Field(default=False)
    root_dir: pathlib.Path = pydantic.Field(default=constants.ROOT)
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR)
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = pydantic.Field(default=constants.OUTPUT_DIR / EXP_NO)
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = pydantic.Field(default=constants.DATA_DIR)
    """data directory. alias to constants.DATA_DIR"""
    device: torch.device = pydantic.Field(default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = pydantic.Field(default=42)

    # -- Train
    train_log_interval: int = pydantic.Field(default=1)
    train_batch_size: int = pydantic.Field(default=32)
    train_n_epochs: int = pydantic.Field(default=50)

    train_use_amp: bool = pydantic.Field(default=True)

    train_loss_name: str = pydantic.Field(default="MSELoss")
    train_loss_params: dict[str, float] = pydantic.Field(default_factory=lambda: {"reduction": "mean"})
    train_optimizer_name: str = pydantic.Field(default="AdamW")
    train_optimizer_params: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-2, "eps": 1e-8, "fused": True}
    )
    train_scheduler_name: str = pydantic.Field(default="CosineLRScheduler")
    train_scheduler_params: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"num_warmup_steps": 1, "num_training_steps": 10, "num_cycles": 0.5, "last_epoch": -1}
    )

    # -- Valid
    valid_batch_size: int = pydantic.Field(default=32)

    # -- Data
    n_folds: int = pydantic.Field(default=5)
    train_data_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "train.csv")
    test_data_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "test.csv")
