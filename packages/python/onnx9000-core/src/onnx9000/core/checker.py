"""ONNX Model and Tensor Checker implementation."""

from typing import Any, Optional

from onnx9000.core.exceptions import UnsupportedOpError, UnsupportedOpsetError, ValidationError
from onnx9000.core.ir import Node, Tensor


class ValidationContext:
    """Context for model validation, tracking errors and configuration."""

    def __init__(
        self,
        strict: bool = True,
        allow_unrecognized_ops: bool = False,
        skip_shape_inference: bool = False,
    ):
        """Initialize the ValidationContext."""
        self.strict = strict
        self.allow_unrecognized_ops = allow_unrecognized_ops
        self.skip_shape_inference = skip_shape_inference
        self.errors: list[str] = []


class SchemaRegistry:
    """Registry for ONNX operator schemas across different domains and opsets."""

    def __init__(self):
        """Initialize the SchemaRegistry with default ONNX schemas."""
        self.schemas = {}
        for i in range(1, 22):
            self.schemas[f"ai.onnx_v{i}"] = {"Conv": {"pads": "ints", "strides": "ints"}}
        for i in range(1, 5):
            self.schemas[f"ai.onnx.ml_v{i}"] = {"TreeEnsembleClassifier": {}}

    def register_custom_schema(self, domain: str, opset: int, schema_json: dict[str, Any]):
        """Register a custom schema for a specific domain and opset."""
        self.schemas[f"{domain}_v{opset}"] = schema_json

    def get_schema(self, op_type: str, opset: int, domain: str = "ai.onnx"):
        """Get the schema for a specific operator."""
        key = f"{domain}_v{opset}"
        if key not in self.schemas:
            raise UnsupportedOpsetError(f"Unsupported opset: {key}")
        if op_type not in self.schemas[key]:
            raise UnsupportedOpError(f"Unsupported op: {op_type} in {key}")
        return self.schemas[key][op_type]


def check_tensor(tensor: Tensor, ctx: ValidationContext):
    """Check a tensor for validity against ONNX rules."""
    if tensor.data_type not in [
        "float",
        "float16",
        "int32",
        "int64",
        "string",
        "bool",
        "float8e4m3fn",
        "float8e5m2",
        "bfloat16",
    ]:
        ctx.errors.append(f"Invalid data_type: {tensor.data_type}")

    for dim in tensor.shape:
        if isinstance(dim, int) and dim < -1:
            ctx.errors.append(f"Invalid dim: {dim}")
        if isinstance(dim, int) and dim == -1 and tensor.is_initializer:
            ctx.errors.append("Initializer cannot have -1 dim")

    # external data checks
    if tensor.data_location == "EXTERNAL":
        if not tensor.external_data:
            ctx.errors.append("External data missing")
        elif ".." in str(tensor.external_data.get("location", "")):
            ctx.errors.append("Directory traversal not allowed in external data")

    # size checks
    if tensor.raw_data:
        calculated_size = 1
        for d in tensor.shape:
            if isinstance(d, int) and d > 0:
                calculated_size *= d
        if len(tensor.raw_data) > 2 * 1024 * 1024 * 1024 and tensor.data_location != "EXTERNAL":
            ctx.errors.append("Tensor exceeds 2GB")


def check_attribute(attr_name: str, attr_val: Any, schema_type: str, ctx: ValidationContext):
    """Check a node attribute for validity."""
    if schema_type == "ints":
        if not isinstance(attr_val, list) or not all(isinstance(x, int) for x in attr_val):
            ctx.errors.append(f"Expected ints for {attr_name}")
    elif schema_type == "floats":
        if not isinstance(attr_val, list) or not all(isinstance(x, (int, float)) for x in attr_val):
            ctx.errors.append(f"Expected floats for {attr_name}")


def _check_op_specific(node: Node, ctx: ValidationContext):
    """Perform operator-specific validity checks."""
    op = node.op_type
    if op in ["Add", "Sub", "Mul", "Div"]:
        if len(node.inputs) != 2:
            ctx.errors.append(f"{op} requires 2 inputs")
    elif op == "Conv":
        if len(node.inputs) < 2:
            ctx.errors.append("Conv requires at least 2 inputs")
        pads = node.attributes.get("pads", [])
        if len(pads) > 0 and len(pads) % 2 != 0:
            ctx.errors.append("Conv pads must be 2 * spatial_dims")
    elif op in ["If", "Loop", "Scan"]:
        if not (
            "then_branch" in node.attributes
            or "else_branch" in node.attributes
            or "body" in node.attributes
        ):
            ctx.errors.append(f"{op} requires subgraph attributes")
    elif op == "TreeEnsembleClassifier":
        if not all(
            k in node.attributes for k in ["nodes_treeids", "nodes_nodeids", "nodes_featureids"]
        ):
            ctx.errors.append(f"{op} missing attributes")


def check_model(model: Any, ctx: Optional[ValidationContext] = None):
    """Check a whole model for validity."""
    ctx = ctx or ValidationContext()

    if getattr(model, "ir_version", 0) < 3 or getattr(model, "ir_version", 0) > 10:
        ctx.errors.append("Invalid ir_version")

    if not isinstance(getattr(model, "producer_name", ""), str):
        ctx.errors.append("Invalid producer_name")

    opset_imports = getattr(model, "opset_import", [])
    if not opset_imports:
        ctx.errors.append("opset_import missing")

    seen_domains = set()
    for imp in opset_imports:
        if imp.domain in seen_domains:
            ctx.errors.append(f"Duplicate domain {imp.domain}")
        seen_domains.add(imp.domain)

    graph = getattr(model, "graph", None)
    if not graph:
        ctx.errors.append("Graph is missing")
        raise ValidationError(", ".join(ctx.errors))

    seen_names = set()
    for i in graph.inputs:
        if i.name in seen_names:
            ctx.errors.append(f"Duplicate input {i.name}")
        seen_names.add(i.name)

    for i in graph.initializers:
        if i.name in seen_names:
            ctx.errors.append(f"Duplicate initializer {i.name}")
        seen_names.add(i.name)
        check_tensor(i, ctx)

    for n in graph.nodes:
        for out in n.outputs:
            if out in seen_names:
                ctx.errors.append(f"Duplicate node output {out}")
            seen_names.add(out)

    for n in graph.nodes:
        _check_op_specific(n, ctx)
        for inp in n.inputs:
            if inp and inp not in seen_names:
                ctx.errors.append(f"Dangling input {inp}")

    if ctx.errors:
        raise ValidationError(", ".join(ctx.errors))

    return True


async def check_model_async(model: Any, ctx: Optional[ValidationContext] = None):
    """Check a whole model for validity asynchronously."""
    return check_model(model, ctx)
