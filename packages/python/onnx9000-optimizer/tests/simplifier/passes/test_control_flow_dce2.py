import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass


def test_loop_unroll_single_scan_out():
    g = Graph("TestLoopUnrollSingle")

    t_m = Tensor("M", (1,), DType.INT64)
    t_m.data = np.array([1])  # Run exactly once
    t_m.is_initializer = True
    g.tensors["M"] = t_m

    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.data = np.array([True])
    t_cond.is_initializer = True
    g.tensors["cond"] = t_cond

    g.tensors["v_in"] = Tensor("v_in", (1,), DType.FLOAT32)

    body_g = Graph("Body")
    body_g.inputs = ["iter", "cond", "v_in"]
    body_g.outputs.append(ValueInfo("cond_out", (1,), DType.BOOL))
    body_g.outputs.append(ValueInfo("v_out", (1,), DType.FLOAT32))
    body_g.outputs.append(ValueInfo("scan_out", (1,), DType.FLOAT32))

    n_abs = Node("Abs", inputs=["v_in"], outputs=["v_out"])
    n_neg = Node("Neg", inputs=["v_in"], outputs=["scan_out"])
    n_cond = Node("Identity", inputs=["cond"], outputs=["cond_out"])
    body_g.nodes.extend([n_abs, n_neg, n_cond])

    # Add an initializer to the subgraph
    t_init = Tensor("sub_init", (1,), DType.FLOAT32)
    t_init.data = np.array([1.0])
    t_init.is_initializer = True
    body_g.tensors["sub_init"] = t_init
    body_g.initializers.append("sub_init")

    n_loop = Node("Loop", inputs=["M", "cond", "v_in"], outputs=["v_final", "scan_final"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)

    cf_pass = ControlFlowFoldingPass()
    changed = cf_pass.run(g)

    assert changed
    ops = [n.op_type for n in g.nodes]
    assert "Unsqueeze" in ops  # Should trigger single scan output branch
    assert "sub_init" not in g.tensors  # Should be mapped to prefixed name
