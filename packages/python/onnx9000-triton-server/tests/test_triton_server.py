"""Comprehensive unit tests for onnx9000.triton_server and onnx9000_triton_server."""

import pytest
from onnx9000.triton_server import (
    KServeV2InferInput,
    TritonModelConfig,
    TritonServer,
    TritonTensorConfig,
)


def test_triton_server_pbtxt_and_kserve_v2() -> None:
    """Verify TritonModelConfig.to_pbtxt(), TritonServer.register_model(), and infer()."""
    t = TritonServer()
    assert t.process("test") == "Triton Server processed test"

    cfg = TritonModelConfig(
        name="resnet50",
        max_batch_size=8,
        inputs=[
            TritonTensorConfig("input_0", "TYPE_FP32", (3, 224, 224)),
            TritonTensorConfig("input_1", "TYPE_FP32", (3, 224, 224)),
        ],
        outputs=[
            TritonTensorConfig("prob", "TYPE_FP32", (1000,)),
            TritonTensorConfig("aux", "TYPE_FP32", (1000,)),
        ],
    )
    pbtxt = t.register_model(cfg)
    assert 'name: "resnet50"' in pbtxt
    assert 'name: "input_0", data_type: TYPE_FP32, dims: [ 3, 224, 224 ]' in pbtxt
    assert 'name: "prob", data_type: TYPE_FP32, dims: [ 1000 ]' in pbtxt

    empty_cfg = TritonModelConfig(name="minimal")
    t.register_model(empty_cfg)
    resp_min = t.infer("minimal", [])
    assert resp_min.outputs[0].name == "output_0"

    req = KServeV2InferInput(name="input_0", shape=[1, 2], datatype="FP32", data=[1.0, 2.0])
    resp = t.infer("resnet50", [req])
    assert resp.model_name == "resnet50"
    assert resp.outputs[0].name == "prob"
    assert resp.outputs[0].data == [1.0, 2.0]

    with pytest.raises(KeyError, match="Unregistered Triton model"):
        t.infer("unknown_model", [])
