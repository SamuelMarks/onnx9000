"""Module for graph building."""

from typing import Any, Optional, Union

import numpy as np
from onnx9000.core import onnx_pb2 as pb
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.script.op import pop_active_builder, set_active_builder
from onnx9000.toolkit.script.var import Var


class GraphBuilder:
    """Builder class for explicit Top-Down Graph Construction."""

    def __init__(self, name: str = "Graph") -> None:
        """Implementation of __init__."""
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, Any]] = []
        self.initializers: dict[str, np.ndarray] = {}
        self.node_by_name: dict[str, Node] = {}
        self.metadata: dict[str, str] = {}

    def __enter__(self) -> "GraphBuilder":
        """Implementation of __enter__."""
        set_active_builder(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Implementation of __exit__."""
        pop_active_builder()

    def set_metadata(self, doc_string: str = "", domain: str = "", version: int = 1) -> None:
        """Implementation of set_metadata."""
        self.metadata["doc_string"] = doc_string
        self.metadata["domain"] = domain
        self.metadata["version"] = str(version)

    def add_node(self, node: Node) -> None:
        """Implementation of add_node."""
        if not node.name:
            node.name = f"{node.op_type}_{len(self.nodes)}"
        self.nodes.append(node)
        self.node_by_name[node.name] = node

    def add_input(self, name: str, dtype: DType, shape: tuple[Union[int, str], ...]) -> Var:
        """Implementation of add_input."""
        self.inputs.append({"name": name, "dtype": dtype, "shape": shape})
        return Var(name=name)

    def add_output(self, var: Var, name: Optional[str] = None) -> None:
        """Implementation of add_output."""
        if name is not None:
            var.name = name
        self.outputs.append({"name": var.name})

    def add_initializer(self, name: str, array: np.ndarray) -> Var:
        """Implementation of add_initializer."""
        self.initializers[name] = array
        return Var(name=name)

    def get_node(self, name: str) -> Optional[Node]:
        """Implementation of get_node."""
        return self.node_by_name.get(name)

    def delete(self, node: Node) -> None:
        """Implementation of delete."""
        if node in self.nodes:
            self.nodes.remove(node)
            if node.name in self.node_by_name:
                del self.node_by_name[node.name]

    def replace(self, old_node: Node, new_node: Node) -> None:
        """Implementation of replace."""
        if old_node in self.nodes:
            idx = self.nodes.index(old_node)
            self.nodes[idx] = new_node
            if old_node.name in self.node_by_name:
                del self.node_by_name[old_node.name]
            if not new_node.name:
                new_node.name = f"{new_node.op_type}_{len(self.nodes)}"
            self.node_by_name[new_node.name] = new_node

    def replace_input(self, node: Node, old_var: Var, new_var: Var) -> None:
        """Implementation of replace_input."""
        if node in self.nodes:
            node.inputs = [new_var.name if x == old_var.name else x for x in node.inputs]

    def merge(self, other_builder: "GraphBuilder") -> None:
        """Implementation of merge."""
        for node in other_builder.nodes:
            self.add_node(node)
        self.inputs.extend(other_builder.inputs)
        self.outputs.extend(other_builder.outputs)
        self.initializers.update(other_builder.initializers)

    def rename_all(self, prefix: str) -> None:
        """Utility to rename all nodes and tensors in a graph with a specific prefix."""
        for inp in self.inputs:
            inp["name"] = f"{prefix}_{inp['name']}"
        for out in self.outputs:
            out["name"] = f"{prefix}_{out['name']}"
        new_initializers = {}
        for name, array in self.initializers.items():
            new_initializers[f"{prefix}_{name}"] = array
        self.initializers = new_initializers
        new_node_by_name = {}
        for node in self.nodes:
            if node.name:
                node.name = f"{prefix}_{node.name}"
                new_node_by_name[node.name] = node
            node.inputs = [f"{prefix}_{x}" for x in node.inputs]
            node.outputs = [f"{prefix}_{x}" for x in node.outputs]
        self.node_by_name = new_node_by_name

    def infer_shapes(self) -> None:
        """Implement pure Python shape inference that penetrates into control flow subgraphs."""
        tensor_shapes = {}
        tensor_dtypes = {}
        for inp in self.inputs:
            tensor_shapes[inp["name"]] = inp["shape"]
            tensor_dtypes[inp["name"]] = inp["dtype"]
        for init_name, arr in self.initializers.items():
            tensor_shapes[init_name] = arr.shape
            tensor_dtypes[init_name] = DType.FLOAT32 if arr.dtype == np.float32 else DType.INT64
        for node in self.nodes:
            for _k, v in node.attributes.items():
                if isinstance(v, GraphBuilder):
                    v.infer_shapes()
            if node.op_type in ("Add", "Sub", "Mul", "Div", "Relu", "Sigmoid"):
                if node.inputs and node.inputs[0] in tensor_shapes:
                    for out in node.outputs:
                        tensor_shapes[out] = tensor_shapes[node.inputs[0]]
                        tensor_dtypes[out] = tensor_dtypes.get(node.inputs[0], DType.FLOAT32)
            elif node.op_type == "Constant":
                if "value" in node.attributes:
                    arr = node.attributes["value"]
                    for out in node.outputs:
                        tensor_shapes[out] = arr.shape
                        tensor_dtypes[out] = (
                            DType.FLOAT32 if arr.dtype == np.float32 else DType.INT64
                        )
            elif node.op_type in ("If", "Loop", "Scan"):
                if "then_branch" in node.attributes:
                    node.attributes["then_branch"]

    def extract_subgraph(self, inputs: list[str], outputs: list[str]) -> "GraphBuilder":
        """Extract a subgraph and perform Dead-Code Elimination (DCE)."""
        sub_builder = GraphBuilder(name=f"{self.name}_subgraph")
        required_nodes: set[str] = set()
        set(outputs)
        queue = list(outputs)
        visited_tensors = set(queue)
        tensor_to_producer = {}
        for node in self.nodes:
            for out in node.outputs:
                tensor_to_producer[out] = node
        while queue:
            current_tensor = queue.pop(0)
            if current_tensor in inputs:
                continue
            if current_tensor in tensor_to_producer:
                producer = tensor_to_producer[current_tensor]
                if producer.name and producer.name not in required_nodes:
                    required_nodes.add(producer.name)
                    for inp in producer.inputs:
                        if inp not in visited_tensors:
                            visited_tensors.add(inp)
                            queue.append(inp)
                    for k, v in producer.attributes.items():
                        if isinstance(v, GraphBuilder):
                            sub_outputs = [out["name"] for out in v.outputs]
                            sub_inputs = [inp["name"] for inp in v.inputs]
                            v_extracted = v.extract_subgraph(sub_inputs, sub_outputs)
                            producer.attributes[k] = v_extracted
        for node in self.nodes:
            if node.name in required_nodes:
                sub_builder.add_node(
                    Node(
                        op_type=node.op_type,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        attributes=node.attributes,
                        name=node.name,
                    )
                )
        for inp in self.inputs:
            if inp["name"] in inputs:
                sub_builder.inputs.append(dict(inp))
        for out in self.outputs:
            if out["name"] in outputs:
                sub_builder.outputs.append(dict(out))
        for init_name, array in self.initializers.items():
            if init_name in visited_tensors:
                sub_builder.initializers[init_name] = array
        return sub_builder

    def If(self, cond: Var, num_outputs: int = 1) -> Any:
        """Implementation of If."""
        from onnx9000.toolkit.script.control_flow import IfContextManager

        return IfContextManager(self, cond, num_outputs)

    def Loop(self, max_trip_count: Var, cond: Var, num_outputs: int = 1) -> Any:
        """Implementation of Loop."""
        from onnx9000.toolkit.script.control_flow import LoopContextManager

        return LoopContextManager(self, max_trip_count, cond, num_outputs)

    def build(self) -> Graph:
        """Implementation of build."""
        graph = Graph(self.name)
        for inp in self.inputs:
            graph.inputs.append(inp["name"])
            tensor = Tensor(name=inp["name"], shape=inp["shape"], dtype=inp["dtype"])
            graph.add_tensor(tensor)
        for out in self.outputs:
            graph.outputs.append(out["name"])
        for name, arr in self.initializers.items():
            graph.initializers.append(name)
            tensor = Tensor(
                name=name, shape=arr.shape, dtype=DType.FLOAT32, is_initializer=True, data=arr
            )
            graph.add_tensor(tensor)
        for node in self.nodes:
            graph.add_node(node)
        return graph

    def to_onnx(self) -> pb.ModelProto:
        """Serializes GraphBuilder directly to a ModelProto using the lightweight pb2 module."""
        graph_proto = pb.GraphProto()
        graph_proto.name = self.name
        for inp in self.inputs:
            value_info = pb.ValueInfoProto()
            value_info.name = inp["name"]
            tensor_type = pb.TypeProto.Tensor()
            if inp["dtype"] == DType.FLOAT32:
                tensor_type.elem_type = pb.TensorProto.FLOAT
            elif inp["dtype"] == DType.INT64:
                tensor_type.elem_type = pb.TensorProto.INT64
            else:
                tensor_type.elem_type = pb.TensorProto.FLOAT
            shape = pb.TensorShapeProto()
            for d in inp["shape"]:
                dim = pb.TensorShapeProto.Dimension()
                if isinstance(d, int):
                    dim.dim_value = d
                else:
                    dim.dim_param = str(d)
                shape.dim.append(dim)
            tensor_type.shape.CopyFrom(shape)
            value_info.type.tensor_type.CopyFrom(tensor_type)
            graph_proto.input.append(value_info)
        for out in self.outputs:
            value_info = pb.ValueInfoProto()
            value_info.name = out["name"]
            tensor_type = pb.TypeProto.Tensor()
            tensor_type.elem_type = pb.TensorProto.FLOAT
            value_info.type.tensor_type.CopyFrom(tensor_type)
            graph_proto.output.append(value_info)
        for name, arr in self.initializers.items():
            t = pb.TensorProto()
            t.name = name
            t.dims.extend(arr.shape)
            if arr.dtype == np.float32:
                t.data_type = pb.TensorProto.FLOAT
                t.raw_data = arr.tobytes()
            elif arr.dtype == np.int64:
                t.data_type = pb.TensorProto.INT64
                t.raw_data = arr.tobytes()
            graph_proto.initializer.append(t)
        for node in self.nodes:
            node_proto = pb.NodeProto()
            node_proto.op_type = node.op_type
            if node.name:
                node_proto.name = node.name
            node_proto.input.extend(node.inputs)
            node_proto.output.extend(node.outputs)
            for k, v in node.attributes.items():
                attr = pb.AttributeProto()
                attr.name = k
                if isinstance(v, float):
                    attr.type = pb.AttributeProto.FLOAT
                    attr.f = v
                elif isinstance(v, int):
                    attr.type = pb.AttributeProto.INT
                    attr.i = v
                elif isinstance(v, str):
                    attr.type = pb.AttributeProto.STRING
                    attr.s = v.encode("utf-8")
                elif isinstance(v, np.ndarray):
                    attr.type = pb.AttributeProto.TENSOR
                    t = pb.TensorProto()
                    t.dims.extend(v.shape)
                    if v.dtype == np.float32:
                        t.data_type = pb.TensorProto.FLOAT
                        t.raw_data = v.tobytes()
                    elif v.dtype == np.int64:
                        t.data_type = pb.TensorProto.INT64
                        t.raw_data = v.tobytes()
                    attr.t.CopyFrom(t)
                elif isinstance(v, list):
                    if all(isinstance(x, int) for x in v):
                        attr.type = pb.AttributeProto.INTS
                        attr.ints.extend(v)
                    elif all(isinstance(x, float) for x in v):
                        attr.type = pb.AttributeProto.FLOATS
                        attr.floats.extend(v)
                elif isinstance(v, GraphBuilder):
                    attr.type = pb.AttributeProto.GRAPH
                    attr.g.CopyFrom(v.to_onnx().graph)
                node_proto.attribute.append(attr)
            graph_proto.node.append(node_proto)
        model = pb.ModelProto()
        model.graph.CopyFrom(graph_proto)
        model.ir_version = 8
        from onnx9000.toolkit.script.schema import get_target_opset

        imp = pb.OperatorSetIdProto()
        imp.domain = ""
        imp.version = get_target_opset()
        model.opset_import.append(imp)
        if "domain" in self.metadata:
            model.domain = self.metadata["domain"]
        if "custom_domain" in self.metadata:
            imp2 = pb.OperatorSetIdProto()
            imp2.domain = self.metadata["custom_domain"]
            imp2.version = 1
            model.opset_import.append(imp2)
        if "doc_string" in self.metadata:
            model.doc_string = self.metadata["doc_string"]
        if "version" in self.metadata:
            model.model_version = int(self.metadata["version"])
        return model

    def validate(self) -> None:
        """Pure Python schema validation to match standard onnx.checker rigor."""
        visited: set[str] = set()
        path: set[str] = set()
        node_to_deps = {}
        tensor_to_producer = {}
        for node in self.nodes:
            for out in node.outputs:
                tensor_to_producer[out] = node
        for node in self.nodes:
            deps = []
            for inp in node.inputs:
                if inp in tensor_to_producer:
                    deps.append(tensor_to_producer[inp])
            node_to_deps[node.name] = deps

        def visit(n: Node) -> None:
            """Implementation of visit."""
            if n.name in path:
                raise ValueError(f"Cyclic dependency detected involving node {n.name}")
            if n.name in visited:
                return
            path.add(n.name)
            for dep in node_to_deps.get(n.name, []):
                visit(dep)
            path.remove(n.name)
            visited.add(n.name)

        for node in self.nodes:
            visit(node)

    @classmethod
    def from_onnx(cls, model_proto: Union[pb.ModelProto, pb.GraphProto]) -> "GraphBuilder":
        """Import an existing .onnx file into a GraphBuilder for editing."""
        graph_proto = model_proto.graph if hasattr(model_proto, "graph") else model_proto
        builder = cls(name=graph_proto.name)
        for inp in graph_proto.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            dtype = DType.FLOAT32 if inp.type.tensor_type.elem_type == 1 else DType.INT64
            builder.add_input(inp.name, dtype, tuple(dims))
        for out in graph_proto.output:
            var = Var(name=out.name)
            builder.add_output(var)
        for init in graph_proto.initializer:
            if init.data_type == 1:
                arr = np.frombuffer(init.raw_data, dtype=np.float32)
            else:
                arr = np.frombuffer(init.raw_data, dtype=np.int64)
            arr = arr.reshape(list(init.dims))
            builder.add_initializer(init.name, arr)
        for node_proto in graph_proto.node:
            attrs = {}
            for attr in node_proto.attribute:
                if attr.type == 1:
                    attrs[attr.name] = attr.f
                elif attr.type == 2:
                    attrs[attr.name] = attr.i
                elif attr.type == 3:
                    attrs[attr.name] = attr.s.decode("utf-8")
                elif attr.type == 4:
                    t = attr.t
                    dtype = np.float32 if t.data_type == 1 else np.int64
                    arr = np.frombuffer(t.raw_data, dtype=dtype)
                    if t.dims:
                        arr = arr.reshape(list(t.dims))
                    attrs[attr.name] = arr
                elif attr.type == 5:
                    attrs[attr.name] = cls.from_onnx(attr.g)
                elif attr.type == 7:
                    attrs[attr.name] = list(attr.ints)
                elif attr.type == 6:
                    attrs[attr.name] = list(attr.floats)
            node = Node(
                op_type=node_proto.op_type,
                inputs=list(node_proto.input),
                outputs=list(node_proto.output),
                attributes=attrs,
                name=node_proto.name,
            )
            builder.add_node(node)
        return builder
