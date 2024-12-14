import pytest

from src import loss


def test_get_loss_fn_should_success() -> None:
    loss_fn = loss.get_loss_fn("BCEWithLogitsLoss", {})
    assert callable(loss_fn), f"E: callable(loss_fn), A: {callable(loss_fn)}"
    assert loss_fn.__class__.__name__ == "BCEWithLogitsLoss", (
        f"E: loss_fn.__class__.__name__ == 'BCEWithLogitsLoss', A: {loss_fn.__class__.__name__}"
    )

    loss_fn = loss.get_loss_fn("CrossEntropyLoss", {})
    assert callable(loss_fn), f"E: callable(loss_fn), A: {callable(loss_fn)}"
    assert loss_fn.__class__.__name__ == "CrossEntropyLoss", (
        f"E: loss_fn.__class__.__name__ == 'CrossEntropyLoss', A: {loss_fn.__class__.__name__}"
    )

    loss_fn = loss.get_loss_fn("MSELoss", {})
    assert callable(loss_fn), f"E: callable(loss_fn), A: {callable(loss_fn)}"
    assert loss_fn.__class__.__name__ == "MSELoss", (
        f"E: loss_fn.__class__.__name__ == 'MSELoss', A: {loss_fn.__class__.__name__}"
    )

    loss_fn = loss.get_loss_fn("L1Loss", {})
    assert callable(loss_fn), f"E: callable(loss_fn), A: {callable(loss_fn)}"
    assert loss_fn.__class__.__name__ == "L1Loss", (
        f"E: loss_fn.__class__.__name__ == 'L1Loss', A: {loss_fn.__class__.__name__}"
    )

    with pytest.raises(ValueError):
        loss_fn = loss.get_loss_fn("UnknownLoss", {})
