"""Caffe to ONNX IR Mapper."""

from typing import Any

import numpy as np
from onnx9000.core.ir import Constant, Graph, Node, Variable


class CaffeMapper:
    """Mapper to convert Caffe layers to ONNX IR."""

    def __init__(self, net_info: dict[str, Any], weights: dict[str, list[np.ndarray]]):
        """Initialize the mapper."""
        self.net_info = net_info
        self.weights = weights
        self.graph = Graph(net_info.get("name", ["CaffeModel"])[0])
        self.tensors = {}

    def get_tensor(self, name: str) -> Variable:
        """Get or create a tensor."""
        if name not in self.tensors:
            t = Variable(name)
            self.tensors[name] = t
            self.graph.add_tensor(t)
        return self.tensors[name]

    def map(self) -> Graph:
        """Map layers to an ONNX IR graph."""
        layers = self.net_info.get("layer", [])
        if not layers:
            layers = self.net_info.get("layers", [])

        # Process input dimensions if they are specified top-level
        input_names = self.net_info.get("input", [])
        input_dims = self.net_info.get("input_dim", [])
        input_shapes = self.net_info.get("input_shape", [])

        if input_names:
            for i, name in enumerate(input_names):
                t = self.get_tensor(name)
                shape = []
                if input_shapes and i < len(input_shapes):
                    dims = input_shapes[i].get("dim", [])
                    shape = [int(d) for d in dims]
                elif input_dims:
                    # legacy input_dim is flat array of [n, c, h, w] for all inputs
                    start = i * 4
                    shape = [int(d) for d in input_dims[start : start + 4]]
                if shape:
                    t.shape = tuple(shape)
                self.graph.inputs.append(t)

        for layer in layers:
            l_type = layer.get("type", [""])[0]
            name = layer.get("name", [""])[0]
            bottoms = layer.get("bottom", [])
            tops = layer.get("top", [])

            if l_type == "Input":
                shape_params = layer.get("input_param", [{}])[0].get("shape", [])
                for i, top in enumerate(tops):
                    t = self.get_tensor(top)
                    if shape_params and i < len(shape_params):
                        dims = shape_params[i].get("dim", [])
                        t.shape = tuple([int(d) for d in dims])
                    if t not in self.graph.inputs:
                        self.graph.inputs.append(t)

            elif l_type == "Data":
                for top in tops:
                    t = self.get_tensor(top)
                    if t not in self.graph.inputs:
                        self.graph.inputs.append(t)

            elif l_type == "Convolution":
                param = layer.get("convolution_param", [{}])[0]
                int(param.get("num_output", [1])[0])
                kernel_size = int(param.get("kernel_size", [1])[0])
                stride = int(param.get("stride", [1])[0])
                pad = int(param.get("pad", [0])[0])
                group = int(param.get("group", [1])[0])
                bias_term = str(param.get("bias_term", ["true"])[0]).lower() != "false"

                inputs = [self.get_tensor(b) for b in bottoms]

                blobs = self.weights.get(name, [])
                if blobs:
                    w = blobs[0]
                    # Caffe weights are already NCHW: (num_output, channels/group, kh, kw)
                    w_t = Constant(f"{name}_w", values=w.tobytes(), shape=w.shape)
                    self.graph.add_tensor(w_t)
                    inputs.append(w_t)
                    if bias_term and len(blobs) > 1:
                        b = blobs[1]
                        b_t = Constant(f"{name}_b", values=b.tobytes(), shape=b.shape)
                        self.graph.add_tensor(b_t)
                        inputs.append(b_t)

                outputs = [self.get_tensor(t) for t in tops]
                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = [kernel_size, kernel_size]
                node.attributes["strides"] = [stride, stride]
                node.attributes["pads"] = [pad, pad, pad, pad]
                node.attributes["group"] = group
                self.graph.add_node(node)

            elif l_type == "InnerProduct":
                param = layer.get("inner_product_param", [{}])[0]
                int(param.get("num_output", [1])[0])
                bias_term = str(param.get("bias_term", ["true"])[0]).lower() != "false"

                inputs = [self.get_tensor(b) for b in bottoms]

                blobs = self.weights.get(name, [])
                if blobs:
                    w = blobs[0]
                    # Caffe inner product weight is (num_output, channels)
                    # ONNX MatMul expects (channels, num_output) if used with MatMul, or we use Gemm.
                    # Gemm handles transB=1.
                    w_t = Constant(f"{name}_w", values=w.tobytes(), shape=w.shape)
                    self.graph.add_tensor(w_t)
                    inputs.append(w_t)

                    if bias_term and len(blobs) > 1:
                        b = blobs[1]
                        b_t = Constant(f"{name}_b", values=b.tobytes(), shape=b.shape)
                        self.graph.add_tensor(b_t)
                        inputs.append(b_t)

                outputs = [self.get_tensor(t) for t in tops]
                # we'll map to Gemm
                node = Node("Gemm", inputs=inputs, outputs=outputs, name=name)
                node.attributes["alpha"] = 1.0
                node.attributes["beta"] = 1.0
                node.attributes["transB"] = 1
                self.graph.add_node(node)

            elif l_type == "Pooling":
                param = layer.get("pooling_param", [{}])[0]
                pool_type = param.get("pool", ["MAX"])[0]
                kernel_size = int(param.get("kernel_size", [1])[0])
                stride = int(param.get("stride", [1])[0])
                pad = int(param.get("pad", [0])[0])

                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]

                op = "AveragePool" if pool_type in ("AVE", 1) else "MaxPool"
                node = Node(op, inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = [kernel_size, kernel_size]
                node.attributes["strides"] = [stride, stride]
                node.attributes["pads"] = [pad, pad, pad, pad]
                self.graph.add_node(node)

            elif l_type == "ReLU":
                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]
                node = Node("Relu", inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)

            elif l_type == "Softmax":
                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]
                node = Node("Softmax", inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)

            else:
                # pass through or generic mapping
                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]
                if inputs and outputs:
                    node = Node(l_type, inputs=inputs, outputs=outputs, name=name)
                    self.graph.add_node(node)

        # Mark last tops as outputs if they are not used as bottoms
        all_bottoms = set()
        for layer in layers:
            all_bottoms.update(layer.get("bottom", []))

        for layer in layers:
            for top in layer.get("top", []):
                if top not in all_bottoms:
                    tensor = self.get_tensor(top)
                    if tensor not in self.graph.outputs:
                        self.graph.outputs.append(tensor)

        return self.graph
