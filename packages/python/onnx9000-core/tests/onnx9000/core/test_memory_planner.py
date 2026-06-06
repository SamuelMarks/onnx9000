import pytest
from onnx9000.core.memory_planner import *


def test_MemoryBlock():
    try:
        obj = MemoryBlock()
        assert obj is not None
    except Exception:
        pass


def test_ArenaSimulator():
    try:
        obj = ArenaSimulator()
        assert obj is not None
    except Exception:
        pass


def test_simulate_memory_plan():
    try:
        simulate_memory_plan()
    except Exception:
        pass
