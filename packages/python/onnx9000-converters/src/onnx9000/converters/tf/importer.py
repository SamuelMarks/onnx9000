"""Module providing core logic and structural definitions."""

from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def _convert_dtype(tf_dtype: int) -> DType:
    """Implements the _convert_dtype method or operation."""
    if tf_dtype == 1:
        return DType.FLOAT32
    if tf_dtype == 3:
        return DType.INT32
    return DType.FLOAT32


class TFImporter:
    """Class TFImporter implementation."""

    def __init__(self) -> None:
        """Implements the __init__ method or operation."""
        self.graph = Graph(name="tf_model")
        self.tensor_map: dict[str, str] = {}

    def convert_nhwc_to_nchw(self, input_name: str, node_name: str) -> str:
        """Implements the convert_nhwc_to_nchw method or operation."""
        t_out = f"{node_name}_nchw"
        t_node = Node(
            op_type="Transpose",
            inputs=[input_name],
            outputs=[t_out],
            attributes={"perm": [0, 3, 1, 2]},
            name=f"{node_name}_transpose_nchw",
        )
        self.graph.nodes.append(t_node)
        self.graph.tensors[t_out] = Tensor(name=t_out, dtype=DType.FLOAT32.value, shape=())
        return t_out

    def convert_nchw_to_nhwc(self, input_name: str, node_name: str) -> str:
        """Implements the convert_nchw_to_nhwc method or operation."""
        t_out = f"{node_name}_nhwc"
        t_node = Node(
            op_type="Transpose",
            inputs=[input_name],
            outputs=[t_out],
            attributes={"perm": [0, 2, 3, 1]},
            name=f"{node_name}_transpose_nhwc",
        )
        self.graph.nodes.append(t_node)
        self.graph.tensors[t_out] = Tensor(name=t_out, dtype=DType.FLOAT32.value, shape=())
        return t_out

    def parse(self, graph_def: dict[str, Any]) -> Graph:
        """Implements the parse method or operation."""
        for node in graph_def.get("node", []):
            op = node["op"]
            name = node["name"]
            inputs = node.get("input", [])
            if op == "Placeholder":
                dt = _convert_dtype(node.get("attr", {}).get("dtype", 1))
                shape = node.get("attr", {}).get("shape", {}).get("dim", [])
                shape_tuple = tuple([d["size"] for d in shape])
                t = Tensor(name=name, dtype=dt.value, shape=shape_tuple)
                self.graph.inputs.append(t)
                self.graph.tensors[name] = t
                self.tensor_map[name] = name
            elif op == "Const":
                dt = _convert_dtype(node.get("attr", {}).get("dtype", 1))
                t = Tensor(name=name, dtype=dt.value, shape=())
                self.graph.initializers.append(name)
                self.graph.tensors[name] = t
                self.tensor_map[name] = name
            elif op == "MatMul":
                out_name = f"{name}:0"
                n = Node(
                    op_type="MatMul",
                    inputs=[self.tensor_map.get(i, i) for i in inputs],
                    outputs=[out_name],
                    attributes={},
                    name=name,
                )
                self.graph.nodes.append(n)
                self.graph.tensors[out_name] = Tensor(
                    name=out_name, dtype=DType.FLOAT32.value, shape=()
                )
                self.tensor_map[name] = out_name
            elif op == "Relu":
                out_name = f"{name}:0"
                n = Node(
                    op_type="Relu",
                    inputs=[self.tensor_map.get(inputs[0], inputs[0])],
                    outputs=[out_name],
                    attributes={},
                    name=name,
                )
                self.graph.nodes.append(n)
                self.graph.tensors[out_name] = Tensor(
                    name=out_name, dtype=DType.FLOAT32.value, shape=()
                )
                self.tensor_map[name] = out_name
            elif op == "Conv2D":
                out_name = f"{name}:0"
                inp = self.tensor_map.get(inputs[0], inputs[0])
                weights = self.tensor_map.get(inputs[1], inputs[1])
                nchw_in = self.convert_nhwc_to_nchw(inp, name)
                w_out = f"{name}_weight_oihw"
                w_node = Node(
                    op_type="Transpose",
                    inputs=[weights],
                    outputs=[w_out],
                    attributes={"perm": [3, 2, 0, 1]},
                    name=f"{name}_weight_trans",
                )
                self.graph.nodes.append(w_node)
                self.graph.tensors[w_out] = Tensor(name=w_out, dtype=DType.FLOAT32.value, shape=())
                conv_out = f"{name}_conv_nchw"
                n = Node(
                    op_type="Conv",
                    inputs=[nchw_in, w_out],
                    outputs=[conv_out],
                    attributes={},
                    name=name,
                )
                self.graph.nodes.append(n)
                self.graph.tensors[conv_out] = Tensor(
                    name=conv_out, dtype=DType.FLOAT32.value, shape=()
                )
                nhwc_out = self.convert_nchw_to_nhwc(conv_out, name)
                id_node = Node(
                    op_type="Identity",
                    inputs=[nhwc_out],
                    outputs=[out_name],
                    attributes={},
                    name=f"{name}_id",
                )
                self.graph.nodes.append(id_node)
                self.graph.tensors[out_name] = Tensor(
                    name=out_name, dtype=DType.FLOAT32.value, shape=()
                )
                self.tensor_map[name] = out_name
            else:
                self.tensor_map[name] = None
        if self.graph.nodes:
            out_name = self.graph.nodes[-1].outputs[0]
            self.graph.outputs.append(self.graph.tensors[out_name])
        return self.graph


def load_tf(model_path_or_dict: Any) -> Graph:
    """Implements the load_tf method or operation."""
    importer = TFImporter()
    if isinstance(model_path_or_dict, dict):
        return importer.parse(model_path_or_dict)
    return importer.graph
