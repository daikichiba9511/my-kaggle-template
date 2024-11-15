import argparse
import multiprocessing as mp
import pathlib
from typing import Any, Callable

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

from . import config, models

logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_loss_fn(loss_name: str, loss_params: dict[str, Any]) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_params)
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**loss_params)
    if loss_name == "MSELoss":
        return nn.MSELoss(**loss_params)
    raise ValueError(f"Unknown loss name: {loss_name}")


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV3,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.Schedulers,
    criterion: LossFn,
    loader: torch_data.DataLoader,
    device: torch.device,
    use_amp: bool,
    scaler: grad_scaler.GradScaler | None = None,
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
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_mode.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            output = model(x)
        y_pred = output
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        ema_model.update(model)

        loss = loss.detach().cpu().item()
        loss_meter.update(loss)
        pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f},Epoch:{epoch}")

    return loss_meter.avg, lr


def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader,
    criterion: LossFn,
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
        score: float
        oof_df: pl.DataFrame
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


# =============================================================================
# Dataset
# =============================================================================
TrainBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]
ValidBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]


class MyTrainDataset(torch_data.Dataset[TrainBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TrainBatch:
        raise NotImplementedError


class MyValidDataset(torch_data.Dataset[ValidBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df

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
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    df = pl.read_csv(df_fp)

    df_train = df.filter(pl.col("fold") != fold)
    df_valid = df.filter(pl.col("fold") == fold)
    assert len(df_train) > 0, f"df_train is empty: {df_fp=}, {fold=}"
    assert len(df_valid) > 0, f"df_valid is empty: {df_fp=}, {fold=}"

    train_ds: torch_data.Dataset[TrainBatch] = MyTrainDataset(df_train)
    valid_ds: torch_data.Dataset[ValidBatch] = MyValidDataset(df_valid)

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


def main() -> None:
    args = parse_args()
    if args.debug:
        cfg = config.Config(is_debug=True)
    else:
        cfg = config.Config()
    utils.pinfo(cfg.model_dump())
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")
    # =============================================================================
    # TrainLoop
    # =============================================================================
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
            cfg.train_data_fp, cfg.train_batch_size, cfg.valid_batch_size, cfg.num_workers, fold
        )
        optimizer = optim.get_optimizer(cfg.train_optimizer_name, cfg.train_optimizer_params, model=model)
        if cfg.train_scheduler_params.get("num_training_steps") == -1:
            scheduler_params = optim.setup_scheduler_params(
                cfg.train_scheduler_params, num_step_per_epoch=len(train_loader), n_epoch=cfg.train_n_epochs
            )
        else:
            scheduler_params = cfg.train_scheduler_params
        scheduler = optim.get_scheduler(cfg.train_scheduler_name, scheduler_params, optimizer=optimizer)
        criterion = get_loss_fn(cfg.train_loss_name, cfg.train_loss_params)
        metrics = train_tools.MetricsMonitor(metrics=["epoch", "train/loss", "lr", "valid/loss", "valid/score"])
        best_score, best_oof = 0.0, pl.DataFrame()
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
            )
            valid_loss_avg, valid_score, valid_oof = valid_one_epoch(
                model=model, loader=valid_loader, criterion=criterion, device=cfg.device
            )
            if valid_score > best_score:
                best_oof = valid_oof
                best_score = valid_score
            metric_map = {
                "epoch": epoch,
                "train/loss": train_loss_avg,
                "lr": lr,
                "valid/loss": valid_loss_avg,
                "valid/score": valid_score,
            }
            metrics.update(metric_map)
            if epoch % cfg.train_log_interval == 0:
                metrics.show()
            if run:
                wandb.log(metric_map)

        # -- Save Results
        best_oof.write_csv(cfg.output_dir / f"oof_{fold}.csv")
        metrics.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)
        model_state = train_tools.get_model_state_dict(ema_model.module)
        save_fp_model = cfg.output_dir / f"last_model_{fold}.pth"
        torch.save(model_state, save_fp_model)
        logger.info(f"Saved model to {save_fp_model}")

        if run is not None:
            run.finish()

        if cfg.is_debug:
            break
    logger.info("End of Training")


if __name__ == "__main__":
    main()
