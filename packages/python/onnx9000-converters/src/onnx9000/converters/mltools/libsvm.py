"""LibSVM parser for pure-Python ONNX conversion."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_libsvm(model_text: str) -> Graph:
    """Parse a LibSVM model format and return an ONNX Graph."""
    lines = model_text.strip().split("\n")

    svm_type = "c_svc"
    kernel_type = "rbf"
    rho = 0.0
    coefs = []

    sv_mode = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if sv_mode:
            parts = line.split()
            coefs.append(float(parts[0]))
        else:
            if line.startswith("svm_type"):
                svm_type = line.split()[1]
            elif line.startswith("kernel_type"):
                kernel_type = line.split()[1]
            elif line.startswith("rho"):
                rho = float(line.split()[1])
            elif line == "SV":
                sv_mode = True

    graph = Graph("LibSVM_Model")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    input_vi = ValueInfo("X", DType.FLOAT32, ["batch_size", 10])
    graph.inputs.append(input_vi)

    out_pred = ValueInfo("Y", DType.FLOAT32, ["batch_size", 1])
    graph.outputs.append(out_pred)

    ktype = kernel_type.upper().encode("utf-8")
    op_type = "SVMRegressor" if "svr" in svm_type else "SVMClassifier"

    attrs = {
        "kernel_type": Attribute(name="kernel_type", attr_type="STRING", value=ktype),
        "rho": Attribute(name="rho", attr_type="FLOATS", value=[rho]),
    }
    if coefs:
        attrs["coefficients"] = Attribute(name="coefficients", attr_type="FLOATS", value=coefs)

    node = Node(
        op_type=op_type,
        inputs=["X"],
        outputs=["Y"],
        attributes=attrs,
        domain="ai.onnx.ml",
    )
    graph.nodes.append(node)
    return graph
