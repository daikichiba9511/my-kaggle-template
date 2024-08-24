import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.utils.data as torch_data

from src import log, metrics, utils


class MyTestDataset(torch_data.Dataset):
    def __init__(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def get_model() -> nn.Module:
    model = ...
    return model  # type: ignore


def get_dataloader(batch_size: int, num_workers: int = 2) -> torch_data.DataLoader:
    ds = MyTestDataset()
    dl = torch_data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )
    return dl


def infer(model: nn.Module, dataloader: torch_data.DataLoader, device: torch.device) -> pl.DataFrame:
    model.eval()
    y_preds = []
    y_trues = []
    for batch in dataloader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            output = model(x)
        y_pred = output
        y_preds.append(y_pred.detach().cpu().numpy())
        if y is not None:
            y_trues.append(y.detach().cpu().numpy())
    output = pl.DataFrame({"y_preds": np.concatenate(y_preds)})
    if y_trues:
        output = output.with_columns(y_trues=pl.Series(np.concatenate(y_trues)))
    return output


def run_valid() -> None:
    logger = log.get_root_logger()
    model = get_model()
    dataloader = get_dataloader(batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = infer(model, dataloader, device)
    score = metrics.score(out["y_preds"].to_numpy(), out["y_trues"].to_numpy())
    logger.info(f"Score: {score}")


if __name__ == "__main__":
    run_valid()
