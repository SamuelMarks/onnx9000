"""
Provides the dynamic op namespace and utility functions for constructing ONNX nodes.
This module manages the active builder context and allows fluent node creation.
"""

import threading
from typing import Any, Union

import numpy as np
from onnx9000.core.ir import Node
from onnx9000.toolkit.script.var import Var

_context = threading.local()


def get_active_builder() -> Any:
    """Retrieves the currently active GraphBuilder from the thread-local context stack."""
    if hasattr(_context, "builder_stack") and _context.builder_stack:
        return _context.builder_stack[-1]
    return None


def set_active_builder(builder: Any) -> None:
    """Pushes a GraphBuilder onto the thread-local context stack, making it active."""
    if not hasattr(_context, "builder_stack"):
        _context.builder_stack = []
    _context.builder_stack.append(builder)


def pop_active_builder() -> None:
    """Removes the currently active GraphBuilder from the thread-local context stack."""
    if hasattr(_context, "builder_stack") and _context.builder_stack:
        _context.builder_stack.pop()


def _make_var(val: Any) -> Var:
    """Implicit casting of python scalars/lists to Constant Vars."""
    if isinstance(val, Var):
        return val
    c_var = Constant(value=val)
    return c_var


def _make_vars(vals: Union[list[Any], tuple[Any, ...]]) -> list[Var]:
    """Converts a collection of Python values into Constant Vars."""
    return [_make_var(v) for v in vals]


class OpNamespace:
    """Dynamic namespace for ONNX operations."""

    def __getattr__(self, op_type: str) -> Any:
        """Returns a generator function for constructing an ONNX node of the specified op_type."""

        def _node_builder(*args: Any, **kwargs: Any) -> Union[Var, tuple[Var, ...]]:
            """Constructs and adds an ONNX node to the currently active GraphBuilder."""
            inputs = []
            for arg in args:
                if isinstance(arg, list) and op_type == "Concat":
                    for a in arg:
                        inputs.append(_make_var(a))
                else:
                    inputs.append(_make_var(arg))
            num_outputs = 1
            if op_type in ["TopK", "Split", "LSTM"] and op_type == "TopK":
                num_outputs = 2
            from onnx9000.toolkit.script.schema import validate_op

            validate_op(op_type, inputs, kwargs)
            if op_type == "Squeeze" and "axes" in kwargs:
                axes_val = kwargs.pop("axes")
                inputs.append(_make_var(axes_val))
            out_vars = [Var() for _ in range(num_outputs)]
            out_names = [v.name for v in out_vars]
            in_names = [v.name for v in inputs]
            node = Node(op_type=op_type, inputs=in_names, outputs=out_names, attributes=kwargs)
            builder = get_active_builder()
            if builder is not None:
                builder.add_node(node)
            if num_outputs == 1:
                return out_vars[0]
            return tuple(out_vars)

        return _node_builder


op = OpNamespace()


def Constant(value: Any) -> Var:
    """Implement op.Constant explicitly."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, float):
        arr = np.array(value, dtype=np.float32)
    elif isinstance(value, int):
        arr = np.array(value, dtype=np.int64)
    elif isinstance(value, list):
        if all(isinstance(x, int) for x in value):
            arr = np.array(value, dtype=np.int64)
        else:
            arr = np.array(value, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported type for Constant: {type(value)}")
    out_var = Var()
    node = Node(op_type="Constant", inputs=[], outputs=[out_var.name], attributes={"value": arr})
    builder = get_active_builder()
    if builder is not None:
        builder.add_node(node)
    return out_var


def If(
    cond: Any, then_branch: Any, else_branch: Any, num_outputs: int = 1
) -> Union[Var, tuple[Var, ...], None]:
    """Builds an ONNX If operation with the given condition and branch subgraphs."""
    cond_var = _make_var(cond)
    out_vars = [Var() for _ in range(num_outputs)]
    out_names = [v.name for v in out_vars]
    node = Node(
        op_type="If",
        inputs=[cond_var.name],
        outputs=out_names,
        attributes={"then_branch": then_branch, "else_branch": else_branch},
    )
    builder = get_active_builder()
    if builder is not None:
        builder.add_node(node)
    if num_outputs == 0:
        return None
    elif num_outputs == 1:
        return out_vars[0]
    return tuple(out_vars)


def Loop(
    max_trip_count: Any, cond: Any, body: Any, num_outputs: int = 1
) -> Union[Var, tuple[Var, ...], None]:
    """Builds an ONNX Loop operation with iteration limits, conditions, and a body subgraph."""
    mtc_var = _make_var(max_trip_count)
    cond_var = _make_var(cond)
    out_vars = [Var() for _ in range(num_outputs)]
    out_names = [v.name for v in out_vars]
    node = Node(
        op_type="Loop",
        inputs=[mtc_var.name, cond_var.name],
        outputs=out_names,
        attributes={"body": body},
    )
    builder = get_active_builder()
    if builder is not None:
        builder.add_node(node)
    if num_outputs == 0:
        return None
    elif num_outputs == 1:
        return out_vars[0]
    return tuple(out_vars)


def Scan(
    body: Any, num_scan_inputs: int, num_outputs: int = 1
) -> Union[Var, tuple[Var, ...], None]:
    """Builds an ONNX Scan operation to iterate a subgraph over one or more input tensors."""
    out_vars = [Var() for _ in range(num_outputs)]
    out_names = [v.name for v in out_vars]
    node = Node(
        op_type="Scan",
        inputs=[],
        outputs=out_names,
        attributes={"body": body, "num_scan_inputs": num_scan_inputs},
    )
    builder = get_active_builder()
    if builder is not None:
        builder.add_node(node)
    if num_outputs == 0:
        return None
    elif num_outputs == 1:
        return out_vars[0]
    return tuple(out_vars)


from onnx9000.toolkit.script.schema import set_target_opset

op.Constant = Constant
op.If = If
op.Loop = Loop
op.Scan = Scan
op.set_target_opset = set_target_opset
