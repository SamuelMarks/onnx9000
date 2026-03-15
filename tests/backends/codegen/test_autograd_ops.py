"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.ops.autograd_ops import (
    generate_relu_grad,
    generate_sgd,
    generate_adamw,
)


def test_autograd_ops():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    t_fwd_in = Tensor("fwd_in", (4,), DType.FLOAT32)
    t_fwd_in.buffer_id = 0
    t_grad_out = Tensor("grad_out", (4,), DType.FLOAT32)
    t_grad_out.buffer_id = 1
    t_grad_in = Tensor("grad_in", (4,), DType.FLOAT32)
    t_grad_in.buffer_id = 2
    t_param = Tensor("param", (4,), DType.FLOAT32)
    t_param.buffer_id = 3
    t_grad = Tensor("grad", (4,), DType.FLOAT32)
    t_grad.buffer_id = 4
    t_m = Tensor("m", (4,), DType.FLOAT32)
    t_m.buffer_id = 5
    t_v = Tensor("v", (4,), DType.FLOAT32)
    t_v.buffer_id = 6
    g.add_tensor(t_fwd_in)
    g.add_tensor(t_grad_out)
    g.add_tensor(t_grad_in)
    g.add_tensor(t_param)
    g.add_tensor(t_grad)
    g.add_tensor(t_m)
    g.add_tensor(t_v)
    ctx = Generator(g)
    node_relu_grad = Node(
        "ReluGrad", inputs=["grad_out", "fwd_in"], outputs=["grad_in"], attributes={}
    )
    code_relu_grad = generate_relu_grad(node_relu_grad, ctx)
    assert "ReluGrad" in code_relu_grad
    assert "_arena[2].resize" in code_relu_grad
    assert "grad_in.data[i]" in code_relu_grad
    node_sgd = Node(
        "SGDOptimizer", inputs=["param", "grad"], outputs=[], attributes={"lr": 0.05}
    )
    code_sgd = generate_sgd(node_sgd, ctx)
    assert "SGDOptimizer" in code_sgd
    assert "0.05" in code_sgd
    assert "param.data[i] -= 0.05f * grad.data[i]" in code_sgd
    node_adamw = Node(
        "AdamWOptimizer",
        inputs=["param", "grad", "m", "v"],
        outputs=[],
        attributes={
            "lr": 0.002,
            "beta1": 0.95,
            "beta2": 0.99,
            "eps": 1e-07,
            "weight_decay": 0.02,
            "step_t": 2.0,
        },
    )
    code_adamw = generate_adamw(node_adamw, ctx)
    assert "AdamWOptimizer" in code_adamw
    assert "0.002" in code_adamw
    assert "0.95" in code_adamw
    assert "0.99" in code_adamw
    assert "1e-07" in code_adamw
    assert "0.02" in code_adamw
    assert "2.0" in code_adamw
    assert "param.data[i] -= 0.002f * 0.02f * param.data[i]" in code_adamw
