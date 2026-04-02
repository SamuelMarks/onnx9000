"""Module providing functionality for test_shape_inference_gap10."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_shape_inference_conv_transpose_gather_pad_missing_inputs():
    """Test shape inference conv transpose gather pad missing inputs."""
    g = Graph("TestGaps")
    n_ct1 = Node("ConvTranspose", inputs=[], outputs=["Y_ct1"])
    g.nodes.append(n_ct1)
    n_ct2 = Node("ConvTranspose", inputs=["missing"], outputs=["Y_ct2"])
    g.nodes.append(n_ct2)
    n_gather1 = Node("Gather", inputs=["X"], outputs=["Y_gather1"])
    g.nodes.append(n_gather1)
    n_gather2 = Node("Gather", inputs=["missing1", "missing2"], outputs=["Y_gather2"])
    g.nodes.append(n_gather2)
    n_pad1 = Node("Pad", inputs=[], outputs=["Y_pad1"])
    g.nodes.append(n_pad1)
    n_pad2 = Node("Pad", inputs=["missing"], outputs=["Y_pad2"])
    g.nodes.append(n_pad2)
    infer_shapes_and_types(g)
    assert "Y_ct1" not in g.tensors
    assert "Y_pad1" not in g.tensors


def test_shape_inference_if_else_graph():
    """Test shape inference if else graph."""
    g = Graph("TestIf")
    cond = Tensor("cond", (), DType.BOOL)
    g.inputs.append("cond")
    g.tensors["cond"] = cond

    then_g = Graph("Then")
    then_g.inputs.append(ValueInfo("then_out", (2, 2), DType.FLOAT32))
    v_then = ValueInfo("v_then_out", (3, 3), DType.INT32)
    then_g.inputs.append(v_then)
    then_g.outputs.extend(["then_out", v_then])
    then_g.tensors["then_out"] = Tensor("then_out", (2, 2), DType.FLOAT32)
    then_g.tensors["v_then_out"] = Tensor("v_then_out", (3, 3), DType.INT32)

    else_g = Graph("Else")
    else_g.inputs.append(ValueInfo("else_out", (2, 2), DType.FLOAT32))
    v_else = ValueInfo("v_else_out", (3, 3), DType.INT32)
    else_g.inputs.append(v_else)
    else_g.outputs.extend(["else_out", v_else])
    else_g.tensors["else_out"] = Tensor("else_out", (2, 2), DType.FLOAT32)
    else_g.tensors["v_else_out"] = Tensor("v_else_out", (3, 3), DType.INT32)

    n_if = Node("If", inputs=["cond"], outputs=["Y_if1", "Y_if2"])
    n_if.attributes["then_branch"] = Attribute("then_branch", None, then_g)
    n_if.attributes["else_branch"] = Attribute("else_branch", None, else_g)
    g.nodes.append(n_if)
    infer_shapes_and_types(g)

    assert g.tensors["Y_if1"].shape == (2, 2)
    assert g.tensors["Y_if2"].shape == (3, 3)

    g2 = Graph("TestIfMismatch")
    cond2 = Tensor("cond2", (), DType.BOOL)
    g2.inputs.append("cond2")
    g2.tensors["cond2"] = cond2

    then_g2 = Graph("Then2")
    then_g2.inputs.append(ValueInfo("o1", (2, 2), DType.FLOAT32))
    then_g2.outputs.append("o1")
    then_g2.tensors["o1"] = Tensor("o1", (2, 2), DType.FLOAT32)

    else_g2 = Graph("Else2")
    else_g2.inputs.append(ValueInfo("o2", (3, 3), DType.FLOAT32))
    else_g2.outputs.append("o2")
    else_g2.tensors["o2"] = Tensor("o2", (3, 3), DType.FLOAT32)

    n_if2 = Node("If", inputs=["cond2"], outputs=["Y_mismatch"])
    n_if2.attributes["then_branch"] = Attribute("then_branch", None, then_g2)
    n_if2.attributes["else_branch"] = Attribute("else_branch", None, else_g2)
    g2.nodes.append(n_if2)

    infer_shapes_and_types(g2)
    assert "Y_mismatch" in g2.tensors


def test_shape_inference_loop_body():
    """Test shape inference loop body."""
    g = Graph("TestLoop")
    g.inputs.append("M")
    g.inputs.append("cond")
    g.inputs.append("v_in")
    g.tensors["M"] = Tensor("M", (), DType.INT64)
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["v_in"] = Tensor("v_in", (2, 2), DType.FLOAT32)

    body_g = Graph("Body")
    v_out = ValueInfo("v_out", (2, 2), DType.FLOAT32)
    body_g.inputs.append(v_out)
    body_g.outputs.append(v_out)
    body_g.tensors["v_out"] = Tensor("v_out", (2, 2), DType.FLOAT32)

    n_loop = Node("Loop", inputs=["M", "cond", "v_in"], outputs=["v_out_loop"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)
    infer_shapes_and_types(g)


def test_shape_inference_ml_domain():
    """Test shape inference ml domain."""
    g = Graph("TestML")
    g.inputs.append("X")
    g.tensors["X"] = Tensor("X", (10, 5), DType.FLOAT32)

    n_afe = Node("ArrayFeatureExtractor", inputs=["X"], outputs=["Y_afe"])
    n_afe.domain = "ai.onnx.ml"
    g.nodes.append(n_afe)

    n_lc = Node("LinearClassifier", inputs=["X"], outputs=["Y_lc1", "Y_lc2"])
    n_lc.domain = "ai.onnx.ml"
    g.nodes.append(n_lc)

    n_unk = Node("UnknownML", inputs=["X"], outputs=["Y_unk"])
    n_unk.domain = "ai.onnx.ml"
    g.nodes.append(n_unk)

    infer_shapes_and_types(g)

    assert "Y_afe" in g.tensors
    assert g.tensors["Y_afe"].shape == (10, 5)
    assert g.tensors["Y_lc1"].shape == (10,)
    assert "Y_unk" in g.tensors


def test_shape_inference_final_gaps():
    """Test shape inference final gaps."""
    g = Graph("TestFinal")

    # Gather missing in2
    g.inputs.append("X")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    n_gather = Node("Gather", inputs=["X", "missing"], outputs=["Y_gather"])
    g.nodes.append(n_gather)

    # Concat axis out of bounds
    g.inputs.append("C1")
    g.tensors["C1"] = Tensor("C1", (5,), DType.FLOAT32)
    n_concat = Node("Concat", inputs=["C1"], outputs=["Y_concat"])
    n_concat.attributes["axis"] = Attribute("axis", value=1)
    g.nodes.append(n_concat)

    n_concat2 = Node("Concat", inputs=["C1"], outputs=["Y_concat2"])
    n_concat2.attributes["axis"] = Attribute("axis", value=1)
    # Make sum_dim dynamic
    g.inputs.append("C2")
    g.tensors["C2"] = Tensor("C2", (5, "dyn"), DType.FLOAT32)
    n_concat2.inputs.append("C2")
    g.nodes.append(n_concat2)

    # Loop with 3 inputs
    g.inputs.append("M")
    g.inputs.append("cond")
    g.inputs.append("v_in")
    g.tensors["M"] = Tensor("M", (), DType.INT64)
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["v_in"] = Tensor("v_in", (2, 2), DType.FLOAT32)

    body_g = Graph("Body")
    v_iter = ValueInfo("iter", (), DType.INT64)
    v_cond = ValueInfo("cond", (), DType.BOOL)
    v_out = ValueInfo("v_out", (1, 1), DType.FLOAT32)
    body_g.inputs.extend([v_iter, v_cond, v_out])
    body_g.outputs.append(v_out)
    body_g.tensors["v_out"] = Tensor("v_out", (2, 2), DType.FLOAT32)

    n_loop = Node("Loop", inputs=["M", "cond", "v_in"], outputs=["v_out_loop"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)

    infer_shapes_and_types(g)
