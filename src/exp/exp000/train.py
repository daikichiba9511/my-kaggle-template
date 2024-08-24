import argparse
import importlib
import multiprocessing as mp
from typing import Any, Callable

import polars as pl
import timm.utils as timm_utils
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import wandb
from torch.cuda import amp
from tqdm.auto import tqdm

from src import constants, log, metrics, optim, train_tools, utils

logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_loss_fn(loss_name: str, loss_params: dict[str, Any]) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_params)
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
) -> tuple[float, float]:
    model = model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("train/loss")
    for _batch_idx, batch in pbar:
        ema_model.update(model)
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp.autocast(enabled=use_amp, dtype=torch.float16):
            output = model(x)
        y_pred = output
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        loss = loss.detach().cpu().item()
        loss_meter.update(loss)
        pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f},Epoch:{epoch}")

    return loss_meter.avg, optimizer.param_groups[0]["lr"]


def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader,
    criterion: LossFn,
    device: torch.device,
) -> tuple[float, float, pl.DataFrame]:
    """
    Returns:
        loss: float
        score: float
        oof_df: pl.DataFrame
    """
    model = model.eval()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Valid", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("valid/loss")
    score_meter = train_tools.AverageMeter("valid/score")
    oof: list[pl.DataFrame] = []
    for batch_idx, batch in pbar:
        x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            output = model(x)

        y_pred = output
        loss = criterion(y_pred.detach().cpu(), y)
        loss_meter.update(loss.item())

        valid_score = metrics.score(y_true=y, y_pred=y_pred)
        score_meter.update(valid_score)

        oof.append(
            train_tools.make_oof(
                x=x.detach().cpu().numpy(),
                y=y.detach().cpu().numpy(),
                y_pred=y_pred.detach().cpu().numpy(),
                x_names=["image_path"],
                y_names=["target"],
                sample_id=None,
            )
        )
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f} AvgScore:{score_meter.avg:.4f}")

    oof_df = pl.concat(oof)
    return loss_meter.avg, score_meter.avg, oof_df


# =============================================================================
# Dataset
# =============================================================================


class MyTrainDataset(torch_data.Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def init_dataloader(
    train_batch_size: int, valid_batch_size: int, num_workers: int = 16
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    # TODO: Implement your own dataset
    train_ds = ...  # type: ignore
    valid_ds = ...  # type: ignore
    assert isinstance(train_ds, torch_data.Dataset)  # type: ignore
    assert isinstance(valid_ds, torch_data.Dataset)  # type: ignore

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
    cfg_module = importlib.import_module(f"src.exp.{EXP_NO}.config")
    models = importlib.import_module(f"src.exp.{EXP_NO}.models")
    if args.debug:
        cfg = cfg_module.Config(is_debug=True)
    else:
        cfg = cfg_module.Config()
    utils.pinfo(cfg.model_dump())
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")
    # =============================================================================
    # TrainLoop
    # =============================================================================
    for fold in range(cfg.n_fold):
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
        model, ema_model = models.compile_models(model, ema_model)
        train_loader, valid_loader = init_dataloader(cfg.train_batch_size, cfg.valid_batch_size, cfg.num_workers)
        optimizer = optim.get_optimizer(cfg.optimizer_name, cfg.optimizer_params, model=model)
        scheduler = optim.get_scheduler(cfg.scheduler_name, cfg.scheduler_params, optimizer=optimizer)
        criterion = get_loss_fn(cfg.train_loss_name, cfg.train_loss_params)
        metrics = train_tools.MetricsMonitor(metrics=["epoch", "train/loss", "lr", "valid/loss", "valid/score"])
        best_score, best_oof = 0.0, pl.DataFrame()
        for epoch in range(cfg.n_epochs):
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
            if run:
                metric_map = {
                    "epoch": epoch,
                    "train/loss": train_loss_avg,
                    "lr": lr,
                    "valid/loss": valid_loss_avg,
                    "valid/score": valid_score,
                }
                wandb.log(metric_map)
                metrics.update(metric_map)
                if epoch % cfg.log_interval == 0:
                    metrics.show()

        # -- Save Results
        best_oof.write_csv(cfg.output_dir / f"oof_{fold}.csv")
        metrics.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)
        torch.save(ema_model.module.state_dict(), cfg.output_dir / f"last_model_{fold}.pth")

        if run is not None:
            run.finish()

        if cfg.is_debug:
            break
    logger.info("End of Training")


if __name__ == "__main__":
    main()
