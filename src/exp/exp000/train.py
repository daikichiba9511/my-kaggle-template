import argparse
import multiprocessing as mp
import pathlib

import numpy as np
import polars as pl
import timm.utils as timm_utils
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import wandb
from torch.amp import autocast_mode, grad_scaler
from tqdm.auto import tqdm
from typing_extensions import TypeAlias

from src import constants, log, metrics, optim, train_tools, utils
from src import loss as my_loss

from . import config, models

logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]
COMMIT_HASH = utils.get_commit_hash_head()
CALLED_TIME = log.get_called_time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


# =============================================================================
# Dataset
# =============================================================================
TrainBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]
ValidBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]


class MyTrainDataset(torch_data.Dataset[TrainBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df.to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TrainBatch:
        raise NotImplementedError


class MyValidDataset(torch_data.Dataset[ValidBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df.to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ValidBatch:
        raise NotImplementedError


def init_dataloader(
    df_fp: pathlib.Path,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int = 16,
    fold: int = 0,
    debug: bool = False,
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    if df_fp.suffix == ".csv":
        df = pl.read_csv(df_fp)
    elif df_fp.suffix == ".parquet":
        df = pl.read_parquet(df_fp)
    else:
        raise ValueError(f"Unknown file type: {df_fp}")

    df_train = df.filter(pl.col("fold") != fold)
    df_valid = df.filter(pl.col("fold") == fold)
    assert len(df_train) > 0, f"df_train is empty: {df_fp=}, {fold=}"
    assert len(df_valid) > 0, f"df_valid is empty: {df_fp=}, {fold=}"
    if debug:
        df_train = df_train.head(100)
        df_valid = df_valid.head(100)
    # --- Preprocess

    # --- Construct Datasets
    train_ds: torch_data.Dataset[TrainBatch] = MyTrainDataset(df_train)
    valid_ds: torch_data.Dataset[ValidBatch] = MyValidDataset(df_valid)
    # --- Construct DataLoaders
    train_dl = torch_data.DataLoader(
        dataset=train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    valid_loader = torch_data.DataLoader(
        dataset=valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_dl, valid_loader


# =============================================================================
# TrainFn
# =============================================================================
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV3,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.Schedulers,
    criterion: my_loss.LossFn,
    loader: torch_data.DataLoader[TrainBatch],
    device: torch.device,
    use_amp: bool = True,
    scaler: grad_scaler.GradScaler | None = None,
    max_norm: float = 1000.0,
) -> tuple[float, float]:
    """
    Args:
        epoch: number of epoch
        model: model to train
        ema_model: timm_utils.ModelEmaV3
        optimizer: torch.optim.Optimizer. I almost use AdamW.
        scheduler: optim.Schedulers. I almost use transformers.get_cosine_schedule_with_warmup
        criterion: LossFn. see get_loss_fn.
        loader: torch_data.DataLoader for training set
        device: torch.device
        use_amp: If True, use auto mixed precision. I use f16 as dtype.
        scaler: grad_scaler.GradScaler | None


    Returns:
        loss_meter.avg: float
        lr : float
    """
    lr = scheduler.get_last_lr()[0]
    model = model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("train/loss")
    for _batch_idx, batch in pbar:
        _sample_id, x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_mode.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            output = model(x)
        y_pred = output
        loss = criterion(y_pred, y)

        # --- Update
        train_tools.step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            max_norm=max_norm,
            scaler=scaler,
            ema_model=ema_model,
            scheduler=scheduler,
        )

        loss_meter.update(loss.detach().cpu().item())
        pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f},Epoch:{epoch},Norm:{grad_norm:.2f}")

    return loss_meter.avg, lr


# =============================================================================
# ValidFn
# =============================================================================
def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader[ValidBatch],
    criterion: my_loss.LossFn,
    device: torch.device,
) -> tuple[float, float, pl.DataFrame]:
    """
    Args:
        model: nn.Module
        loader: torch_data.DataLoader for validation set
        criterion: LossFn
        device: torch.device

    Returns: tuple
        loss: float
        valid_score: float
        oof: pl.DataFrame
    """
    model = model.eval()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Valid", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("valid/loss")
    oofs: list[pl.DataFrame] = []
    for batch_idx, batch in pbar:
        sample_id, x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            output = model(x)

        y_pred = output
        loss = criterion(y_pred.detach().cpu(), y)
        loss_meter.update(loss.item())

        oofs.append(
            pl.DataFrame({
                "sample_id": sample_id,
                "y": y.cpu().detach().numpy(),
                "y_pred": y_pred.cpu().detach().numpy(),
            })
        )
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f}")

    oof = pl.concat(oofs)
    valid_score = metrics.score(y_true=oof["y"].to_numpy(), y_pred=oof["y_pred"].to_numpy())
    return loss_meter.avg, valid_score, oof


def main() -> None:
    args = parse_args()
    if args.debug:
        cfg = config.Config(is_debug=True)
    else:
        cfg = config.Config()
    utils.pinfo(cfg.model_dump())
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")
    # =============================================================================
    # TrainLoop
    # =============================================================================
    score_folds, oof_folds = [], []
    for fold in range(cfg.n_folds):
        logger.info(f"Start fold: {fold}")
        utils.seed_everything(cfg.seed + fold)
        if cfg.is_debug:
            run = None
        else:
            run = wandb.init(
                project=constants.COMPE_NAME,
                name=f"{cfg.name}_{fold}",
                config=cfg.model_dump(),
                reinit=True,
                group=f"{fold}",
                dir="./src",
            )
        model, ema_model = models.get_model(cfg.model_name, cfg.model_params)
        if args.compile:
            model, ema_model = models.compile_models(model, ema_model)
        model, ema_model = model.to(cfg.device), ema_model.to(cfg.device)
        train_loader, valid_loader = init_dataloader(
            df_fp=cfg.train_data_fp,
            train_batch_size=cfg.train_batch_size,
            valid_batch_size=cfg.valid_batch_size,
            num_workers=cfg.num_workers,
            fold=fold,
            debug=cfg.is_debug,
        )
        optimizer = optim.get_optimizer(cfg.train_optimizer_name, cfg.train_optimizer_params, model=model)
        if cfg.train_scheduler_params.get("num_training_steps") == -1:
            scheduler_params = optim.setup_scheduler_params(
                cfg.train_scheduler_params, num_step_per_epoch=len(train_loader), n_epoch=cfg.train_n_epochs
            )
        else:
            scheduler_params = cfg.train_scheduler_params
        scheduler = optim.get_scheduler(cfg.train_scheduler_name, scheduler_params, optimizer=optimizer)
        criterion = my_loss.get_loss_fn(cfg.train_loss_name, cfg.train_loss_params)
        metric_monitor = train_tools.MetricsMonitor(metrics=["epoch", "train/loss", "lr", "valid/loss", "valid/score"])
        best_score, best_oof = 0.0 if cfg.train_is_maximize else float("inf"), pl.DataFrame()
        for epoch in range(cfg.train_n_epochs):
            train_loss_avg, lr = train_one_epoch(
                epoch=epoch,
                model=model,
                ema_model=ema_model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=cfg.device,
                use_amp=cfg.train_use_amp,
                max_norm=cfg.train_max_norm,
            )
            valid_loss_avg, valid_score, valid_oof = valid_one_epoch(
                model=ema_model.module, loader=valid_loader, criterion=criterion, device=cfg.device
            )
            is_enable_to_update = (
                (cfg.save_last_oof and epoch == cfg.train_n_epochs - 1)  # Save last oof
                or (valid_score > best_score if cfg.train_is_maximize else valid_score < best_score)  # Save best oof
            )
            if is_enable_to_update:
                best_oof = valid_oof
                best_score = valid_score

            metric_map = {
                "epoch": epoch,
                "train/loss": train_loss_avg,
                "lr": lr,
                "valid/loss": valid_loss_avg,
                "valid/score": valid_score,
            }
            metric_monitor.update(metric_map)
            if epoch % cfg.train_log_interval == 0:
                metric_monitor.show(use_logger=epoch == cfg.train_n_epochs - 1)
            if run:
                wandb.log(metric_map)

        # -- Save Results
        score_folds.append(best_score)
        oof_folds.append(best_oof)

        best_oof.write_csv(cfg.output_dir / f"oof_{fold}.csv")
        metric_monitor.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)
        model_state = train_tools.get_model_state_dict(ema_model.module)
        save_fp_model = cfg.output_dir / f"last_model_{fold}.pth"
        torch.save(model_state, save_fp_model)
        logger.info(f"Saved model to {save_fp_model}")

        if run is not None:
            run.finish()

        if cfg.is_debug:
            break

    oof = pl.concat(oof_folds)
    score_all = metrics.score(y_true=oof["y"].to_numpy(), y_pred=oof["y_pred"].to_numpy())

    logger.info(f"""\n

===============================================================
End of Training. NAME: {cfg.name},
DESC: {cfg.description}

Scores: {score_folds}, Mean: {np.mean(score_folds)} +/- {np.std(score_folds)}
Score Whold Fold: {score_all}

{CALLED_TIME=}, DURATION={log.calc_duration_from(CALLED_TIME)}, {COMMIT_HASH=}
===============================================================

    \n""")


if __name__ == "__main__":
    main()
