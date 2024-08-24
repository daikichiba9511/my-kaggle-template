import numpy.typing as npt
import torch


def score(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
) -> float:
    raise NotImplementedError
