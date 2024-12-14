import pytest
import torch
import torch.nn as nn

from src import engine


@pytest.fixture(scope="module")
def sample_model() -> nn.Module:
    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.fc = nn.Linear(128, 10)
            self.bn = nn.BatchNorm2d(128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.bn(x)
            x = self.fc(x)
            return x

    model = TestModel()
    return model


def test_AveragaMeter_should_success() -> None:
    avg_meter = engine.AverageMeter("test")

    avg_meter.update(1)
    avg_meter.update(2)
    avg_meter.update(3)

    assert avg_meter.avg == 2.0, f"E: avg_meter.avg == 2.0, A: avg_meter.avg == {avg_meter.avg}"

    avg_meter_dict = avg_meter.to_dict()

    # Check keys
    assert avg_meter_dict["name"] == "test", (
        f"E: avg_meter_dict['name'] == 'test', A: avg_meter_dict['name'] == {avg_meter_dict['name']}"
    )
    assert avg_meter_dict["avg"] == 2.0, (
        f"E: avg_meter_dict['avg'] == 2.0, A: avg_meter_dict['avg'] == {avg_meter_dict['avg']}"
    )
    assert avg_meter_dict["raw_values"] == [1, 2, 3], (
        f"E: avg_meter_dict['raw_values'] == [1, 2, 3], A: avg_meter_dict['raw_values'] == {avg_meter_dict['raw_values']}"
    )


def test_get_model_state_dict_should_success(sample_model: nn.Module) -> None:
    model_state_dict = engine.get_model_state_dict(sample_model)
    assert isinstance(model_state_dict, dict), f"E: isinstance(model_state_dict, dict), A: {type(model_state_dict)}"


def test_UpdateManager_should_success() -> None:
    update_manager = engine.UpdateManager(is_maximize=True, n_epochs=10)

    assert not update_manager.check_epoch(1), "E: update_manager.check_epoch(1) is False, A: True"
    assert update_manager.check_epoch(10), "E: update_manager.check_epoch(10) is True, A: False"
    assert update_manager.check_epoch(11), "E: update_manager.check_epoch(11) is True, A: False"

    update_manager = engine.UpdateManager(is_maximize=True, n_epochs=10)
    update_manager.check_score(0.1)
    update_manager.check_score(0.2)
    update_manager.check_score(0.3)
    update_manager.check_score(0.1)
    update_manager.check_score(0.2)
    assert update_manager.best_score == 0.3, f"E: update_manager.best_score == 0.3, A: {update_manager.best_score}"

    update_manager = engine.UpdateManager(is_maximize=False, n_epochs=10)
    update_manager.check_score(0.1)
    update_manager.check_score(0.2)
    update_manager.check_score(0.3)
    update_manager.check_score(0.1)
    update_manager.check_score(0.2)
    assert update_manager.best_score == 0.1, f"E: update_manager.best_score == 0.3, A: {update_manager.best_score}"
