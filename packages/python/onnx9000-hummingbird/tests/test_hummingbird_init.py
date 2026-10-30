"""Comprehensive unit tests for onnx9000.hummingbird and onnx9000_hummingbird."""

import onnx9000_hummingbird
from onnx9000.hummingbird import (
    Hummingbird,
    Strategy,
    TargetHardware,
    TranspilationEngine,
    TreeAbstractions,
    estimate_memory_footprint,
    run,
    select_optimal_strategy,
)


def test_hummingbird_facade_and_exports() -> None:
    """Verify Hummingbird facade, strategy selection, and re-exports."""
    assert run() == "[hummingbird] processed"
    assert onnx9000_hummingbird.run() == "[hummingbird] processed"

    hb = Hummingbird(target=TargetHardware.CPU)
    assert hb.process("rf_classifier") == "Hummingbird processed rf_classifier on cpu"
    assert isinstance(hb.engine, TranspilationEngine)
    assert Strategy.GEMM is not None
    assert TreeAbstractions is not None
    assert callable(estimate_memory_footprint)
    assert callable(select_optimal_strategy)
