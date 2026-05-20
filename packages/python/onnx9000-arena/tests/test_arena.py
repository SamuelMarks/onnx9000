import pytest
from onnx9000_arena import plan


def test_plan():
    assert plan("model") == "[Arena] planner processed model"


def test_plan_invalid():
    with pytest.raises(ValueError):
        plan("")
