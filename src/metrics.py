import torch


def score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    raise NotImplementedError
