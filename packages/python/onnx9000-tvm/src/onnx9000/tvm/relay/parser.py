import json
from typing import Any, Dict, List

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from .ty import FuncType, TensorType, TupleType, Type


class IRSpy:
    def __init__(self):
        self.node_id_map: dict[int, int] = {}
        self.nodes: list[dict[str, Any]] = []

    def get_id(self, expr: Any) -> int:
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
        elif isinstance(expr, Constant):
            # Try to serialize data
            try:
                import numpy as np

                if isinstance(expr.data, np.ndarray):
                    data_rep = expr.data.tolist()
                else:
                    data_rep = expr.data
            except:
                data_rep = expr.data

            self.nodes.append(
                {
                    "type": "Constant",
                    "data": data_rep,
                    "type_annotation": self.serialize_type(expr.type_annotation),
                }
            )
        elif isinstance(expr, Op):
            self.nodes.append({"type": "Op", "name": expr.name})
        elif isinstance(expr, Call):
            op_id = self.get_id(expr.op)
            args_ids = [self.get_id(arg) for arg in expr.args]
            self.nodes.append({"type": "Call", "op": op_id, "args": args_ids, "attrs": expr.attrs})
        elif isinstance(expr, TupleExpr):
            fields_ids = [self.get_id(f) for f in expr.fields]
            self.nodes.append({"type": "Tuple", "fields": fields_ids})
        elif isinstance(expr, TupleGetItem):
            tuple_id = self.get_id(expr.tuple_value)
            self.nodes.append(
                {"type": "TupleGetItem", "tuple_value": tuple_id, "index": expr.index}
            )
        elif isinstance(expr, Let):
            var_id = self.get_id(expr.var)
            val_id = self.get_id(expr.value)
            body_id = self.get_id(expr.body)
            self.nodes.append({"type": "Let", "var": var_id, "value": val_id, "body": body_id})
        elif isinstance(expr, If):
            cond_id = self.get_id(expr.cond)
            t_id = self.get_id(expr.true_branch)
            f_id = self.get_id(expr.false_branch)
            self.nodes.append(
                {"type": "If", "cond": cond_id, "true_branch": t_id, "false_branch": f_id}
            )
        elif isinstance(expr, Function):
            params_ids = [self.get_id(p) for p in expr.params]
            body_id = self.get_id(expr.body)
            self.nodes.append(
                {
                    "type": "Function",
                    "params": params_ids,
                    "body": body_id,
                    "ret_type": self.serialize_type(expr.ret_type),
                }
            )
        else:
            raise ValueError(f"Unknown expr {type(expr)}")

        return node_id

    def serialize_type(self, ty: Type) -> Any:
        if ty is None:
            return None
        if isinstance(ty, TensorType):
            return {"type": "TensorType", "shape": ty.shape, "dtype": ty.dtype}
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
    spy = IRSpy()
    root_id = spy.get_id(expr)
    return json.dumps({"root": root_id, "nodes": spy.nodes})


def load_json(json_str: str) -> Expr:
    data = json.loads(json_str)
    nodes = data["nodes"]
    root_id = data["root"]

    parsed_nodes: dict[int, Any] = {}

    def parse_type(ty_data: Any) -> Type:
        if not ty_data:
            return None
        t = ty_data["type"]
        if t == "TensorType":
            return TensorType(tuple(ty_data["shape"]), ty_data["dtype"])
        if t == "TupleType":
            return TupleType([parse_type(f) for f in ty_data["fields"]])
        if t == "FuncType":
            return FuncType(
                [parse_type(a) for a in ty_data["arg_types"]], parse_type(ty_data["ret_type"])
            )
        return None

    def get_node(nid: int) -> Any:
        if nid in parsed_nodes:
            return parsed_nodes[nid]

        n = nodes[nid]
        t = n["type"]

        if t == "Var":
            res = Var(name_hint=n["name"], type_annotation=parse_type(n.get("type_annotation")))
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
            res = Let(var=get_node(n["var"]), value=get_node(n["value"]), body=get_node(n["body"]))
        elif t == "If":
            res = If(
                cond=get_node(n["cond"]),
                true_branch=get_node(n["true_branch"]),
                false_branch=get_node(n["false_branch"]),
            )
        elif t == "Function":
            res = Function(
                params=[get_node(p) for p in n["params"]],
                body=get_node(n["body"]),
                ret_type=parse_type(n.get("ret_type")),
            )
        else:
            raise ValueError(f"Unknown type {t}")

        parsed_nodes[nid] = res
        return res

    return get_node(root_id)
