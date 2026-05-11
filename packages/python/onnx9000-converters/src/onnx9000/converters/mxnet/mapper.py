"""MXNet to ONNX IR Mapper."""

import ast
from typing import Any

import numpy as np
from onnx9000.core.ir import Constant, Graph, Node, Variable


class MXNetMapper:
    """Mapper to convert MXNet parsed models to ONNX IR."""

    def __init__(self, symbol_info: dict[str, Any], weights: dict[str, np.ndarray]):
        """Initialize mapper."""
        self.symbol_info = symbol_info
        self.weights = weights
        self.graph = Graph("MXNetModel")
        self.tensors = {}

    def get_tensor(self, name: str) -> Variable:
        """Get or create tensor."""
        if name not in self.tensors:
            t = Variable(name)
            self.tensors[name] = t
            self.graph.add_tensor(t)
        return self.tensors[name]

    def map(self) -> Graph:
        """Map MXNet model to ONNX IR."""
        nodes = self.symbol_info.get("nodes", [])

        for i, node_info in enumerate(nodes):
            op_type = node_info.get("op", "")
            name = node_info.get("name", "")

            if op_type == "null":
                # Variable/Input or weight
                t = self.get_tensor(name)
                # If it's not in weights, it's an input
                if name not in self.weights:
                    self.graph.inputs.append(t)
                else:
                    w = self.weights[name]
                    c = Constant(name, values=w.tobytes(), shape=w.shape)
                    self.tensors[name] = c
                    self.graph.add_tensor(c)
                continue

            inputs = []
            for inp in node_info.get("inputs", []):
                # inp is typically [node_idx, output_idx, 0]
                idx = inp[0]
                inp_name = nodes[idx]["name"]
                inputs.append(self.get_tensor(inp_name))

            t_out = self.get_tensor(name)
            outputs = [t_out]

            # Param attributes usually stored in 'attrs' or 'param'
            attrs = node_info.get("attrs", node_info.get("param", {}))

            if op_type == "Convolution":
                kernel = ast.literal_eval(attrs.get("kernel", "(1, 1)"))
                stride = ast.literal_eval(attrs.get("stride", "(1, 1)"))
                pad = ast.literal_eval(attrs.get("pad", "(0, 0)"))

                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = list(kernel)
                node.attributes["strides"] = list(stride)
                node.attributes["pads"] = [pad[0], pad[1], pad[0], pad[1]]
                self.graph.add_node(node)

            elif op_type == "Pooling":
                pool_type = attrs.get("pool_type", "max")
                kernel = ast.literal_eval(attrs.get("kernel", "(1, 1)"))
                stride = ast.literal_eval(attrs.get("stride", "(1, 1)"))
                pad = ast.literal_eval(attrs.get("pad", "(0, 0)"))

                op = "MaxPool" if pool_type == "max" else "AveragePool"
                node = Node(op, inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = list(kernel)
                node.attributes["strides"] = list(stride)
                node.attributes["pads"] = [pad[0], pad[1], pad[0], pad[1]]
                self.graph.add_node(node)

            elif op_type == "Activation":
                act_type = attrs.get("act_type", "relu")
                if act_type == "relu":
                    node = Node("Relu", inputs=inputs, outputs=outputs, name=name)
                    self.graph.add_node(node)

            elif op_type == "FullyConnected":
                node = Node("Gemm", inputs=inputs, outputs=outputs, name=name)
                node.attributes["alpha"] = 1.0
                node.attributes["beta"] = 1.0
                node.attributes["transB"] = 1
                self.graph.add_node(node)

            else:
                node = Node(op_type, inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)

        # Heads
        heads = self.symbol_info.get("heads", [])
        for h in heads:
            idx = h[0]
            out_name = nodes[idx]["name"]
            t = self.get_tensor(out_name)
            if t not in self.graph.outputs:
                self.graph.outputs.append(t)

        return self.graph
