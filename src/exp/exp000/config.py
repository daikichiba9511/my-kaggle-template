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
    name: str = EXP_NO
    description: str = DESCRIPTION

    # -- General
    is_debug: bool = False
    root_dir: pathlib.Path = constants.ROOT
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = constants.INPUT_DIR
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / EXP_NO
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = constants.DATA_DIR
    """data directory. alias to constants.DATA_DIR"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0 if is_debug else 16
    seed: int = 42

    # -- Train
    train_log_interval: int = 1
    train_batch_size: int = 32
    train_n_epochs: int = 10

    train_use_amp: bool = True

    train_loss_name: str = "MSELoss"
    train_loss_params: dict[str, str] = {"reduction": "mean"}
    train_optimizer_name: str = "AdamW"
    train_optimizer_params: dict[str, float] = {"lr": 1e-3, "weight_decay": 1e-2, "eps": 1e-8, "fused": True}
    train_scheduler_name: str = "CosineLRScheduler"
    train_scheduler_params: dict[str, float] = {
        "num_warmup_steps": 1,
        "num_training_steps": 10,
        "num_cycles": 0.5,
        "last_epoch": -1,
    }

    # -- Valid
    valid_batch_size: int = 32

    # -- Data
    n_folds: int = 5
    train_data_fp: pathlib.Path = constants.DATA_DIR / "train.csv"
    test_data_fp: pathlib.Path = constants.DATA_DIR / "test.csv"

    # -- Model
    model_name: str = "SimpleNN"
    model_params: dict[str, float] = {}
