import numpy as np

from src import utils


def test_to_heatmap_should_success() -> None:
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    heatmap = utils.to_heatmap(x=x, y=y)
    assert heatmap.shape == (3, 3), f"E: heatmap.shape == (3, 3), A: {heatmap.shape}"
    assert np.all(heatmap == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), (
        f"E: np.all(heatmap == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), A: {heatmap}"
    )
