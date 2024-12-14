import pytest
import torch
import torch.nn as nn

from src import optim


@pytest.fixture(scope="module")
def sample_model() -> nn.Module:
    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.fc = nn.Linear(128, 10)
            self.ln = nn.LayerNorm(128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.ln(x)
            x = self.fc(x)
            return x

    model = TestModel()
    return model


def test_get_params_no_decay_should_success(sample_model: nn.Module) -> None:
    no_decay = ("bias", "ln.weight", "ln.bias")
    params = optim.get_params_no_decay(sample_model, weight_decay=0.01, no_decay=no_decay)
    assert isinstance(params, list), f"E: isinstance(params, list), A: {type(params)}"
    assert len(params) == 2, f"E: len(params) == 2, A: {len(params)}"
    assert isinstance(params[0], dict), f"E: isinstance(params[0], dict), A: {type(params[0])}"
    assert isinstance(params[1], dict), f"E: isinstance(params[1], dict), A: {type(params[1])}"

    # Check no_decay params and decay params
    # param_keys: ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc.weight', 'fc.bias', 'ln.weight', 'ln.bias']
    # no_decay: ('bias', 'ln.weight', 'ln.bias')
    #
    # decay_params_keys: ('conv1.weight', 'conv2.weight', 'fc.weight')
    # no_decay_params_keys: ('conv1.bias', 'conv2.bias', 'fc.bias', 'ln.weight', 'ln.bias')
    decay_params = params[0]["params"]
    assert len(decay_params) == 3, f"E: len(decay_params) == 3, A: {len(decay_params)}"
    no_decay_params = params[1]["params"]
    assert len(no_decay_params) == 5, f"E: len(no_decay_params) == 5, A: {len(no_decay_params)}"


def test_get_optimizer_should_success(sample_model: nn.Module) -> None:
    optimizer = optim.get_optimizer(
        optimizer_name="AdamW",
        optmizer_params={"lr": 0.001, "weight_decay": 1e-5, "eps": 1e-6, "fused": False},
        model=sample_model,
    )
    assert isinstance(optimizer, torch.optim.Optimizer), (
        f"E: isinstance(optimizer, torch.optim.Optimizer), A: {type(optimizer)}"
    )
    assert optimizer.__class__.__name__ == "AdamW", (
        f"E: optimizer.__class__.__name__ == 'AdamW', A: {optimizer.__class__.__name__}"
    )


def test_setup_scheduler_params_should_success() -> None:
    scheduler_params = {
        "num_warmup_steps": 1,
        "num_training_steps": -1,
        "num_cycles": 0.5,
        "last_epoch": -1,
    }
    len_dataloader = 100
    n_epoch = 10
    warmup_epochs = 1
    # When num_training_steps is -1, automatically set to num_step_per_epoch * n_epoch
    scheduler_params = optim.setup_scheduler_params(
        scheduler_params=scheduler_params,
        num_step_per_epoch=len_dataloader,
        n_epoch=n_epoch,
        warmup_epochs=warmup_epochs,
    )
    assert scheduler_params["num_training_steps"] == len_dataloader * n_epoch, (
        f"E: scheduler_params['num_training_steps'] == len_dataloader * n_epoch, "
        f"A: {scheduler_params['num_training_steps']}"
    )
