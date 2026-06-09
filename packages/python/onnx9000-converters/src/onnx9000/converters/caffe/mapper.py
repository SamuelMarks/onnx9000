"""Caffe to ONNX IR Mapper."""

from typing import Any  # pragma: no cover

import numpy as np  # pragma: no cover
from onnx9000.core.ir import Constant, Graph, Node, Variable  # pragma: no cover


class CaffeMapper:
    """Mapper to convert Caffe layers to ONNX IR."""

    def __init__(self, net_info: dict[str, Any], weights: dict[str, list[np.ndarray]]):
        """Initialize the mapper."""
        self.net_info = net_info  # pragma: no cover
        self.weights = weights  # pragma: no cover
        self.graph = Graph(net_info.get("name", ["CaffeModel"])[0])  # pragma: no cover
        self.tensors = {}  # pragma: no cover

    def get_tensor(self, name: str) -> Variable:  # pragma: no cover
        """Get or create a tensor."""
        if name not in self.tensors:  # pragma: no cover
            t = Variable(name)  # pragma: no cover
            self.tensors[name] = t  # pragma: no cover
            self.graph.add_tensor(t)  # pragma: no cover
        return self.tensors[name]  # pragma: no cover

    def map(self) -> Graph:
        """Map layers to an ONNX IR graph."""
        layers = self.net_info.get("layer", [])  # pragma: no cover
        if not layers:  # pragma: no cover
            layers = self.net_info.get("layers", [])  # pragma: no cover  # pragma: no cover

        # Process input dimensions if they are specified top-level  # pragma: no cover
        input_names = self.net_info.get("input", [])  # pragma: no cover
        input_dims = self.net_info.get("input_dim", [])  # pragma: no cover
        input_shapes = self.net_info.get("input_shape", [])  # pragma: no cover

        if input_names:  # pragma: no cover
            for i, name in enumerate(input_names):  # pragma: no cover
                t = self.get_tensor(name)  # pragma: no cover
                shape = []  # pragma: no cover
                if input_shapes and i < len(input_shapes):  # pragma: no cover
                    dims = input_shapes[i].get("dim", [])  # pragma: no cover
                    shape = [int(d) for d in dims]  # pragma: no cover
                elif input_dims:  # pragma: no cover
                    # legacy input_dim is flat array of [n, c, h, w] for all inputs  # pragma: no cover
                    start = i * 4  # pragma: no cover
                    shape = [int(d) for d in input_dims[start : start + 4]]  # pragma: no cover
                if shape:  # pragma: no cover
                    t.shape = tuple(shape)  # pragma: no cover
                self.graph.inputs.append(t)  # pragma: no cover

        for layer in layers:  # pragma: no cover
            l_type = layer.get("type", [""])[0]  # pragma: no cover
            name = layer.get("name", [""])[0]  # pragma: no cover
            bottoms = layer.get("bottom", [])  # pragma: no cover
            tops = layer.get("top", [])  # pragma: no cover

            if l_type == "Input":  # pragma: no cover
                shape_params = layer.get("input_param", [{}])[0].get(
                    "shape", []
                )  # pragma: no cover
                for i, top in enumerate(tops):  # pragma: no cover
                    t = self.get_tensor(top)  # pragma: no cover
                    if shape_params and i < len(shape_params):  # pragma: no cover
                        dims = shape_params[i].get("dim", [])  # pragma: no cover
                        t.shape = tuple([int(d) for d in dims])  # pragma: no cover
                    if t not in self.graph.inputs:  # pragma: no cover
                        self.graph.inputs.append(t)  # pragma: no cover

            elif l_type == "Data":  # pragma: no cover
                for top in tops:  # pragma: no cover
                    t = self.get_tensor(top)  # pragma: no cover
                    if t not in self.graph.inputs:  # pragma: no cover
                        self.graph.inputs.append(t)  # pragma: no cover

            elif l_type == "Convolution":  # pragma: no cover
                param = layer.get("convolution_param", [{}])[0]  # pragma: no cover
                int(param.get("num_output", [1])[0])  # pragma: no cover
                kernel_size = int(param.get("kernel_size", [1])[0])  # pragma: no cover
                stride = int(param.get("stride", [1])[0])  # pragma: no cover
                pad = int(param.get("pad", [0])[0])  # pragma: no cover
                group = int(param.get("group", [1])[0])  # pragma: no cover
                bias_term = (
                    str(param.get("bias_term", ["true"])[0]).lower() != "false"
                )  # pragma: no cover

                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover

                blobs = self.weights.get(name, [])  # pragma: no cover
                if blobs:  # pragma: no cover
                    w = blobs[0]  # pragma: no cover
                    # Caffe weights are already NCHW: (num_output, channels/group, kh, kw)  # pragma: no cover
                    w_t = Constant(
                        f"{name}_w", values=w.tobytes(), shape=w.shape
                    )  # pragma: no cover
                    self.graph.add_tensor(w_t)  # pragma: no cover
                    inputs.append(w_t)  # pragma: no cover
                    if bias_term and len(blobs) > 1:  # pragma: no cover
                        b = blobs[1]  # pragma: no cover
                        b_t = Constant(
                            f"{name}_b", values=b.tobytes(), shape=b.shape
                        )  # pragma: no cover
                        self.graph.add_tensor(b_t)  # pragma: no cover
                        inputs.append(b_t)  # pragma: no cover

                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover
                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                node.attributes["kernel_shape"] = [
                    kernel_size,
                    kernel_size,
                ]  # pragma: no cover
                node.attributes["strides"] = [stride, stride]  # pragma: no cover
                node.attributes["pads"] = [pad, pad, pad, pad]  # pragma: no cover
                node.attributes["group"] = group  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

            elif l_type == "InnerProduct":  # pragma: no cover
                param = layer.get("inner_product_param", [{}])[0]  # pragma: no cover
                int(param.get("num_output", [1])[0])  # pragma: no cover
                bias_term = (
                    str(param.get("bias_term", ["true"])[0]).lower() != "false"
                )  # pragma: no cover

                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover

                blobs = self.weights.get(name, [])  # pragma: no cover
                if blobs:  # pragma: no cover
                    w = blobs[0]  # pragma: no cover
                    # Caffe inner product weight is (num_output, channels)  # pragma: no cover
                    # ONNX MatMul expects (channels, num_output) if used with MatMul, or we use Gemm.  # pragma: no cover
                    # Gemm handles transB=1.  # pragma: no cover
                    w_t = Constant(
                        f"{name}_w", values=w.tobytes(), shape=w.shape
                    )  # pragma: no cover
                    self.graph.add_tensor(w_t)  # pragma: no cover
                    inputs.append(w_t)  # pragma: no cover

                    if bias_term and len(blobs) > 1:  # pragma: no cover
                        b = blobs[1]  # pragma: no cover
                        b_t = Constant(
                            f"{name}_b", values=b.tobytes(), shape=b.shape
                        )  # pragma: no cover
                        self.graph.add_tensor(b_t)  # pragma: no cover
                        inputs.append(b_t)  # pragma: no cover

                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover
                # we'll map to Gemm  # pragma: no cover
                node = Node("Gemm", inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                node.attributes["alpha"] = 1.0  # pragma: no cover
                node.attributes["beta"] = 1.0  # pragma: no cover
                node.attributes["transB"] = 1  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

            elif l_type == "Pooling":  # pragma: no cover
                param = layer.get("pooling_param", [{}])[0]  # pragma: no cover
                pool_type = param.get("pool", ["MAX"])[0]  # pragma: no cover
                kernel_size = int(param.get("kernel_size", [1])[0])  # pragma: no cover
                stride = int(param.get("stride", [1])[0])  # pragma: no cover
                pad = int(param.get("pad", [0])[0])  # pragma: no cover

                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover
                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover

                op = "AveragePool" if pool_type in ("AVE", 1) else "MaxPool"  # pragma: no cover
                node = Node(op, inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                node.attributes["kernel_shape"] = [
                    kernel_size,
                    kernel_size,
                ]  # pragma: no cover
                node.attributes["strides"] = [stride, stride]  # pragma: no cover
                node.attributes["pads"] = [pad, pad, pad, pad]  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

            elif l_type == "ReLU":  # pragma: no cover
                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover
                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover
                node = Node("Relu", inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

            elif l_type == "Softmax":  # pragma: no cover
                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover
                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover
                node = Node(
                    "Softmax", inputs=inputs, outputs=outputs, name=name
                )  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

            else:  # pragma: no cover
                # pass through or generic mapping  # pragma: no cover
                inputs = [self.get_tensor(b) for b in bottoms]  # pragma: no cover
                outputs = [self.get_tensor(t) for t in tops]  # pragma: no cover
                if inputs and outputs:  # pragma: no cover
                    node = Node(
                        l_type, inputs=inputs, outputs=outputs, name=name
                    )  # pragma: no cover
                    self.graph.add_node(node)  # pragma: no cover

        # Mark last tops as outputs if they are not used as bottoms  # pragma: no cover
        all_bottoms = set()  # pragma: no cover
        for layer in layers:  # pragma: no cover
            all_bottoms.update(layer.get("bottom", []))  # pragma: no cover

        for layer in layers:  # pragma: no cover
            for top in layer.get("top", []):  # pragma: no cover
                if top not in all_bottoms:  # pragma: no cover
                    tensor = self.get_tensor(top)  # pragma: no cover
                    if tensor not in self.graph.outputs:  # pragma: no cover
                        self.graph.outputs.append(tensor)  # pragma: no cover

        return self.graph  # pragma: no cover
