import pytest
from onnx9000.converters.frontend.quantization import (
    AQTParser,
    AWQParser,
    GGUFQuantizationMapper,
    GPTQParser,
)
from onnx9000.core.ir import Graph


def test_gguf_mapper():
    g = Graph("test")
    mapper = GGUFQuantizationMapper()
    mapper.map_block(g, "Q4_K_M", "w1", 32)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "DequantizeLinear"
    assert g.nodes[0].attributes["block_size"].value == 32
    assert g.nodes[0].attributes["gguf_type"].value == "Q4_K_M"


def test_awq_parser():
    g = Graph("test")
    parser = AWQParser()
    parser.parse_config(g, {"group_size": 64}, "w1")
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "DequantizeLinear"
    assert g.nodes[0].attributes["block_size"].value == 64


def test_gptq_parser():
    g = Graph("test")
    parser = GPTQParser()
    parser.parse_state_dict(g, {"w1.g_idx": [1, 0]}, "w1")
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Gather"


def test_aqt_parser():
    g = Graph("test")
    parser = AQTParser()
    parser.parse_aqt(g, "w1", 4)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "DequantizeLinear"
    assert g.nodes[0].attributes["symmetric"].value == 1
    assert g.nodes[0].attributes["bitwidth"].value == 4
