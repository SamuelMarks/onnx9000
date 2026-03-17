import pytest
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.surgeon import merge_lora_adapters


def test_merge_lora_adapters():
    g = Graph("g")
    g.add_tensor(Tensor("master_weight", (10, 10), "float32", is_initializer=True))
    g.initializers.append("master_weight")
    g.add_tensor(Tensor("lora_a_master", (2, 10), "float32", is_initializer=True))
    g.initializers.append("lora_a_master")
    g.add_tensor(Tensor("lora_b_master", (10, 2), "float32", is_initializer=True))
    g.initializers.append("lora_b_master")

    merge_lora_adapters(g)
    assert len(g.initializers) > 0
