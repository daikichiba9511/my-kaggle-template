import argparse

import polars as pl
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import wandb
from timm.utils import model_ema
from torch.amp import autocast_mode, grad_scaler
from tqdm.auto import tqdm

from src import constants, engine, log, metrics, optim, utils
from src import loss as my_loss

from . import config, dataset, models

logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]
COMMIT_HASH = utils.get_commit_hash_head()
CALLED_TIME = log.get_called_time()

# --- Set Environment
# 固定サイズに対して最適なアルゴリズムを選択するようにする
torch.backends.cudnn.benchmark = True
# default:highestで最大精度を指定してるが、速度重視のためmediumにしておく
torch.set_float32_matmul_precision("medium")


# =============================================================================
# TrainFn
# =============================================================================
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: model_ema.ModelEmaV3 | None,
    optimizer: torch.optim.optimizer.Optimizer,
    scheduler: optim.Schedulers,
    criterion: my_loss.LossFn,
    loader: torch_data.DataLoader[dataset.TrainBatch],
    device: torch.device,
    use_amp: bool = True,
    scaler: grad_scaler.GradScaler | None = None,
    max_norm: float = 1000.0,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    """
    Args:
        epoch: number of epoch
        model: model to train
        ema_model: ema_model.ModelEmaV3
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
    lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad()
    model = model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", dynamic_ncols=True)
    loss_meter = engine.AverageMeter("train/loss")
    update_per_epoch = (len(loader) + grad_accum_steps - 1) // grad_accum_steps
    num_updates = update_per_epoch * epoch

    for step, batch in pbar:
        _sample_id, x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_mode.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            output = model(x)
            y_pred = output
            loss = criterion(y_pred, y)

        # --- Update
        if step % grad_accum_steps == 0:
            num_updates += 1
        grad_norm = engine.step(
            step=step,
            model=model,
            optimizer=optimizer,
            loss=loss,
            max_norm=max_norm,
            scaler=scaler,
            ema_model=ema_model,
            scheduler=scheduler,
            grad_accum_steps=grad_accum_steps,
            num_updates=num_updates,
        )

        loss_meter.update(loss.detach().cpu().item())
        pbar.set_postfix_str(f"Epoch {epoch}: Loss={loss_meter.avg:.4f}, Norm={grad_norm:.2f}")

    train_results = {"train/loss": loss_meter.avg, "lr": lr}
    return train_results


# =============================================================================
# ValidFn
# =============================================================================
def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader[dataset.ValidBatch],
    criterion: my_loss.LossFn,
    device: torch.device,
) -> tuple[dict[str, float], pl.DataFrame]:
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
    loss_meter = engine.AverageMeter("valid/loss")
    oofs: list[pl.DataFrame] = []
    for batch_idx, batch in pbar:
        sample_id, x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            output = model(x)

        y_pred = output
        loss = criterion(y_pred, y)
        loss_meter.update(loss.detach().cpu().item())

        oofs.append(
            pl.DataFrame({
                "sample_id": sample_id,
                "y": y.detach().cpu().numpy(),
                "y_pred": y_pred.detach().cpu().numpy(),
            })
        )
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss={loss_meter.avg:.4f}")

    oof = pl.concat(oofs) if len(oofs) > 0 else pl.DataFrame()
    valid_score = metrics.score(y_true=oof["y"].to_numpy(), y_pred=oof["y_pred"].to_numpy())
    valid_results = {"valid/loss": loss_meter.avg, "valid/score": valid_score}
    return valid_results, oof


def main(
    debug: bool = False,
    compile: bool = False,
    seed: int = 42,
    compile_mode: str = "max-autotune",
    compile_dynamic: bool = False,
    fulltrain: bool = False,
) -> None:
    cfg = config.Config(is_debug=debug, seed=seed)
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))

    utils.pinfo(cfg.model_dump())
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")

    # =============================================================================
    # TrainLoop
    # =============================================================================
    score_folds, oof_folds = [], []
    for fold in range(cfg.n_folds):
        logger.info(f"Start fold: {fold}")
        utils.seed_everything(cfg.seed + fold)
        run = (
            wandb.init(
                project=constants.COMPE_NAME,
                name=f"{cfg.name}_{fold}" + "fulltrain" if fulltrain else "",
                config=cfg.model_dump(),
                reinit=True,
                group=f"{fold}",
                settings=wandb.Settings(save_code=True, code_dir="./src"),
            )
            if not cfg.is_debug
            else None
        )
        model, ema_model = models.get_model(cfg.model_arch, cfg.model_params, cfg.use_ema, cfg.ema_params)
        if compile:
            model, ema_model = models.compile_models(model, ema_model, compile_mode, compile_dynamic)
        model = model.to(cfg.device, non_blocking=True)
        ema_model = ema_model.to(cfg.device, non_blocking=True) if ema_model is not None else None

        train_loader, valid_loader = dataset.init_dataloader(
            df_fp=cfg.data_fp_train,
            train_batch_size=cfg.train_batch_size,
            valid_batch_size=cfg.valid_batch_size,
            num_workers=cfg.num_workers,
            fold=fold,
            debug=cfg.is_debug,
            fulltrain=fulltrain,
        )

        optimizer = optim.get_optimizer(cfg.train_optimizer_name, cfg.train_optimizer_params, model=model)
        if cfg.train_scheduler_params.get("num_training_steps") == -1:
            scheduler_params = optim.setup_scheduler_params(
                cfg.train_scheduler_params, num_step_per_epoch=len(train_loader), n_epoch=cfg.train_n_epochs
            )
        else:
            scheduler_params = cfg.train_scheduler_params
        scheduler = optim.get_scheduler(cfg.train_scheduler_name, scheduler_params, optimizer=optimizer)
        scaler = grad_scaler.GradScaler(enabled=cfg.train_use_amp)

        criterion = my_loss.get_loss_fn(cfg.train_loss_name, cfg.train_loss_params)

        metric_monitor = engine.MetricsMonitor(["epoch", "train/loss", "lr", "valid/loss", "valid/score"])
        update_manager = engine.UpdateManager(is_maximize=cfg.train_is_maximize, n_epochs=cfg.train_n_epochs)

        oof_fold = pl.DataFrame()
        for epoch in range(cfg.train_n_epochs):
            train_results = train_one_epoch(
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
                grad_accum_steps=cfg.train_grad_accum_steps,
                scaler=scaler,
            )
            if fulltrain:
                logger.info("Fulltrain mode. Skip validation.")
                metric_map = {
                    "epoch": epoch,
                    **train_results,
                }
                metric_monitor.update(metric_map)
                if epoch % cfg.train_log_interval == 0:
                    metric_monitor.show(use_logger=epoch == cfg.train_n_epochs - 1)
                if run:
                    wandb.log(metric_map)
                continue

            valid_results, valid_oof = valid_one_epoch(
                model=model if ema_model is None else ema_model.module,
                loader=valid_loader,
                criterion=criterion,
                device=cfg.device,
            )
            # --- save last epoch: 基本的にはearly stoppingしない
            if not cfg.train_save_best_model and update_manager.is_last_epoch(epoch):
                oof_fold = valid_oof

            if cfg.train_save_best_model and update_manager.check_score_updated(valid_results["valid/score"]):
                model_state = engine.get_model_state_dict(ema_model.module if ema_model is not None else model)
                save_fp_model = cfg.output_dir / f"best_model_{fold}.pt"
                torch.save(model_state, save_fp_model)

            metric_map = {
                "epoch": epoch,
                **train_results,
                **valid_results,
            }
            metric_monitor.update(metric_map)
            if epoch % cfg.train_log_interval == 0:
                metric_monitor.show(use_logger=epoch == cfg.train_n_epochs - 1)
            if run:
                wandb.log(metric_map)

            if update_manager.is_early_stopping:
                logger.info(f"Detected early stopping. Break training loop. Best score: {update_manager.best_score}")
                break

            # -- End of epoch

        # -- Save Results
        score_folds.append(update_manager.best_score)
        oof_folds.append(oof_fold)

        oof_fold.write_parquet(cfg.output_dir / f"oof_{fold}.parquet")
        metric_monitor.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)

        model_state = engine.get_model_state_dict(ema_model.module if ema_model is not None else model)
        save_fp_model = cfg.output_dir / f"last_model_{fold}.pt"
        torch.save(model_state, save_fp_model)
        logger.info(f"Saved model to {save_fp_model}")

        if run is not None:
            run.finish()
        if cfg.is_debug:
            break

        # -- End of fold
    oof = pl.concat(oof_folds)
    oof.write_parquet(cfg.output_dir / "oof.parquet")
    log.info_stats(cfg=cfg.model_dump(), scores=score_folds, called_time=CALLED_TIME, commit_hash=COMMIT_HASH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile_mode", type=str, default="max-autotune", choices=["default", "max-autotune"])
    parser.add_argument("--compile_dynamic", action="store_true", default=False)
    parser.add_argument("--fulltrain", action="store_true", default=False)
    args = parser.parse_args()
    main(
        debug=args.debug,
        compile=args.compile,
        seed=args.seed,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        fulltrain=args.fulltrain,
    )
