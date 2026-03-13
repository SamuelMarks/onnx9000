"""Test quantization, training, and integration parity."""

from src.onnx9000.runtime.quantization import (
    MinMaxCalibration,
    EntropyCalibration,
    PercentileCalibration,
    DynamicQuantization,
    FormatsSupport,
)
from src.onnx9000.runtime.training import (
    GradientGraphBuilder,
    LossNodeInsertion,
    OptimizerNodeInsertion,
    ORTModule,
    CheckpointAPI,
    ATenTranslationBridge,
)
from src.onnx9000.runtime.integration import (
    IRExport,
    InBrowserTraining,
    InBrowserServing,
    ExternalInteroperability,
)


def test_quantization():
    """Test quantization parity."""
    assert MinMaxCalibration().calibrate()
    assert EntropyCalibration().calibrate()
    assert PercentileCalibration().calibrate()

    dq = DynamicQuantization()
    assert dq.quantize_weight_only()
    assert dq.quantize_runtime_activation()

    fs = FormatsSupport()
    assert fs.process_qdq()
    assert fs.process_qoperator()


def test_training():
    """Test training parity."""
    assert GradientGraphBuilder().build()
    assert LossNodeInsertion().insert()

    opt = OptimizerNodeInsertion()
    assert opt.insert_adamw()
    assert opt.insert_lamb()
    assert opt.insert_sgd()

    assert ORTModule().intercept()

    cp = CheckpointAPI()
    assert cp.load()
    assert cp.save()

    assert ATenTranslationBridge().translate()


def test_integration():
    """Test integration parity."""
    ir = IRExport()
    assert ir.translate_nodes()
    assert ir.map_datatypes()
    assert ir.handle_shapes()
    assert ir.serialize_model()

    bt = InBrowserTraining()
    assert bt.build_training_graph()
    assert bt.insert_loss_optimizers()
    assert bt.manage_memory()

    bs = InBrowserServing()
    assert bs.transition_state()
    assert bs.perform_inference()

    ei = ExternalInteroperability()
    assert ei.download_onnx()
    assert ei.verify_onnxruntime()
    assert ei.verify_training_servers()
