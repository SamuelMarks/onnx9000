"""Relay IR JSON serializer and deserializer."""

from __future__ import annotations

import json

from onnx9000.core.exceptions import CompilationError

from .expr import (
    Call,
    Constant,
    Expr,
    Function,
    If,
    Let,
    Op,
    TupleExpr,
    TupleGetItem,
    Var,
)
from .ty import FuncType, TensorType, TupleType, Type

JsonScalar = str | int | float | bool | None
JsonValue = (
    JsonScalar
    | list[JsonScalar]
    | tuple[JsonScalar, ...]
    | dict[str, JsonScalar | list[JsonScalar]]
)


class IRSpy:
    """Serializes Relay AST nodes into a JSON-compatible table."""

    def __init__(self) -> None:
        """Initialize the serializer node table."""
        self.node_id_map: dict[int, int] = {}
        self.nodes: list[dict[str, JsonValue]] = []

    def get_id(self, expr: Expr) -> int:
        """Serialize an expression node and return its table index."""
        if id(expr) in self.node_id_map:
            return self.node_id_map[id(expr)]

        node_id = len(self.nodes)
        self.node_id_map[id(expr)] = node_id

        if isinstance(expr, Var):
            self.nodes.append(
                {
                    "type": "Var",
                    "name": expr.name_hint,
                    "type_annotation": self.serialize_type(expr.type_annotation),
                }
            )
            return node_id
        if isinstance(expr, Constant):
            try:
                import numpy as np

                if isinstance(expr.data, np.ndarray):
                    data_rep: JsonValue = expr.data.tolist()
                else:
                    data_rep = expr.data  # type: ignore[assignment]
            except ValueError:
                data_rep = expr.data  # type: ignore[assignment]

            self.nodes.append(
                {
                    "type": "Constant",
                    "data": data_rep,
                    "type_annotation": self.serialize_type(expr.type_annotation),
                }
            )
            return node_id
        if isinstance(expr, Op):
            self.nodes.append({"type": "Op", "name": expr.name})
            return len(self.nodes) - 1
        if isinstance(expr, Call):
            op_id = self.get_id(expr.op)
            args_ids: list[JsonValue] = [self.get_id(arg) for arg in expr.args]
            attrs_val: JsonValue = expr.attrs  # type: ignore[assignment]
            self.nodes.append({"type": "Call", "op": op_id, "args": args_ids, "attrs": attrs_val})
            return len(self.nodes) - 1
        if isinstance(expr, TupleExpr):
            fields_ids: list[JsonValue] = [self.get_id(f) for f in expr.fields]
            self.nodes.append({"type": "Tuple", "fields": fields_ids})
            return len(self.nodes) - 1
        if isinstance(expr, TupleGetItem):
            tuple_id = self.get_id(expr.tuple_value)
            self.nodes.append(
                {"type": "TupleGetItem", "tuple_value": tuple_id, "index": expr.index}
            )
            return len(self.nodes) - 1
        if isinstance(expr, Let):
            var_id = self.get_id(expr.var)
            val_id = self.get_id(expr.value)
            body_id = self.get_id(expr.body)
            self.nodes.append({"type": "Let", "var": var_id, "value": val_id, "body": body_id})
            return len(self.nodes) - 1
        if isinstance(expr, If):
            cond_id = self.get_id(expr.cond)
            t_id = self.get_id(expr.true_branch)
            f_id = self.get_id(expr.false_branch)
            self.nodes.append(
                {"type": "If", "cond": cond_id, "true_branch": t_id, "false_branch": f_id}
            )
            return len(self.nodes) - 1
        if isinstance(expr, Function):
            params_ids: list[JsonValue] = [self.get_id(p) for p in expr.params]
            body_id = self.get_id(expr.body)
            self.nodes.append(
                {
                    "type": "Function",
                    "params": params_ids,
                    "body": body_id,
                    "ret_type": self.serialize_type(expr.ret_type),
                }
            )
            return len(self.nodes) - 1
        raise CompilationError(f"Unknown expr {type(expr)}")

    def serialize_type(self, ty: Type | None) -> JsonValue:
        """Serialize a Relay Type into a JSON dictionary."""
        if ty is None:
            return None
        if isinstance(ty, TensorType):
            return {"type": "TensorType", "shape": list(ty.shape), "dtype": ty.dtype}
        if isinstance(ty, TupleType):
            return {"type": "TupleType", "fields": [self.serialize_type(f) for f in ty.fields]}
        if isinstance(ty, FuncType):
            return {
                "type": "FuncType",
                "arg_types": [self.serialize_type(a) for a in ty.arg_types],
                "ret_type": self.serialize_type(ty.ret_type),
            }
        return None


def save_json(expr: Expr) -> str:
    """Serialize a Relay expression tree into a JSON string."""
    spy = IRSpy()
    root_id = spy.get_id(expr)
    return json.dumps({"root": root_id, "nodes": spy.nodes})


def load_json(json_str: str) -> Expr:
    """Deserialize a Relay expression tree from a JSON string."""
    data = json.loads(json_str)
    nodes = data["nodes"]
    root_id = int(data["root"])

    parsed_nodes: dict[int, Expr] = {}

    def parse_type(ty_data: dict[str, JsonValue] | None) -> Type | None:
        """Parse a serialized Type dictionary."""
        if not ty_data:
            return None
        t = ty_data["type"]
        if t == "TensorType":
            return TensorType(tuple(ty_data["shape"]), str(ty_data["dtype"]))  # type: ignore[arg-type]
        if t == "TupleType":
            return TupleType([parse_type(f) for f in ty_data["fields"]])  # type: ignore[misc,union-attr]
        if t == "FuncType":
            return FuncType(
                [parse_type(a) for a in ty_data["arg_types"]],  # type: ignore[misc,union-attr]
                parse_type(ty_data["ret_type"]),  # type: ignore[arg-type]
            )
        return None

    def get_node(nid: int) -> Expr:
        """Reconstruct a node from its serialized representation."""
        if nid in parsed_nodes:
            return parsed_nodes[nid]

        n = nodes[nid]
        t = n["type"]

        if t == "Var":
            res: Expr = Var(
                name_hint=n["name"], type_annotation=parse_type(n.get("type_annotation"))
            )
        elif t == "Constant":
            res = Constant(data=n["data"], type_annotation=parse_type(n.get("type_annotation")))
        elif t == "Op":
            res = Op(name=n["name"])
        elif t == "Call":
            op = get_node(n["op"])
            args = [get_node(a) for a in n["args"]]
            res = Call(op=op, args=args, attrs=n.get("attrs"))
        elif t == "Tuple":
            res = TupleExpr(fields=[get_node(f) for f in n["fields"]])
        elif t == "TupleGetItem":
            res = TupleGetItem(tuple_value=get_node(n["tuple_value"]), index=n["index"])
        elif t == "Let":
            res = Let(
                var=get_node(n["var"]),  # type: ignore[arg-type]
                value=get_node(n["value"]),
                body=get_node(n["body"]),
            )
        elif t == "If":
            res = If(
                cond=get_node(n["cond"]),
                true_branch=get_node(n["true_branch"]),
                false_branch=get_node(n["false_branch"]),
            )
        elif t == "Function":
            res = Function(
                params=[get_node(p) for p in n["params"]],  # type: ignore[misc]
                body=get_node(n["body"]),
                ret_type=parse_type(n.get("ret_type")),
            )
        else:
            raise CompilationError(f"Unknown type {t}")

        parsed_nodes[nid] = res
        return res

    return get_node(root_id)
