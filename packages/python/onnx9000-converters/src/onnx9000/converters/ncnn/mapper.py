"""NCNN to ONNX IR Mapper."""

from typing import Any

from onnx9000.converters.ncnn.weights import WeightsReader
from onnx9000.core.ir import Constant, Graph, Node, Variable


class NCNNMapper:
    """Mapper to convert NCNN parsed layers and weights to ONNX IR."""

    def __init__(self, param_info: dict[str, Any], weights_reader: WeightsReader):
        """Initialize the mapper.

        Args:
            param_info: Parsed param file info.
            weights_reader: Weights reader.
        """
        self.param_info = param_info
        self.weights_reader = weights_reader
        self.graph = Graph("NCNNModel")
        self.tensors = {}

    def get_tensor(self, name: str) -> Variable:
        """Get or create a tensor by name."""
        if name not in self.tensors:
            t = Variable(name)
            self.tensors[name] = t
            self.graph.add_tensor(t)
        return self.tensors[name]

    def map(self) -> Graph:
        """Map layers to an ONNX IR graph."""
        layers = self.param_info.get("layers", [])

        for i, layer in enumerate(layers):
            l_type = layer["type"]
            name = layer["name"]
            bottoms = layer["bottoms"]
            tops = layer["tops"]
            params = layer["params"]

            if l_type == "Input":
                w = params.get(0, 224)
                h = params.get(1, 224)
                c = params.get(2, 3)
                for top in tops:
                    t = self.get_tensor(top)
                    t.shape = (1, c, h, w)
                    self.graph.inputs.append(t)

            elif l_type == "Convolution" or l_type == "ConvolutionDepthWise":
                num_output = params.get(0, 1)
                kernel_w = params.get(1, 1)
                kernel_h = params.get(11, kernel_w)
                stride_w = params.get(3, 1)
                stride_h = params.get(13, stride_w)
                pad_left = params.get(4, 0)
                pad_right = params.get(15, pad_left)
                pad_top = params.get(14, pad_left)
                pad_bottom = params.get(16, pad_top)
                bias_term = params.get(5, 0)
                weight_data_size = params.get(6, 0)
                group = params.get(7, 1)
                if l_type == "ConvolutionDepthWise":
                    group = num_output

                inputs = [self.get_tensor(b) for b in bottoms]

                weight_data = self.weights_reader.read_blob(weight_data_size)
                # Reshape weight depending on group
                # For basic mapping, we just add it as Constant
                w_t = Constant(f"{name}_w", values=weight_data.tobytes())
                self.graph.add_tensor(w_t)
                inputs.append(w_t)

                if bias_term:
                    bias_data = self.weights_reader.read_blob(num_output)
                    b_t = Constant(f"{name}_b", values=bias_data.tobytes())
                    self.graph.add_tensor(b_t)
                    inputs.append(b_t)

                outputs = [self.get_tensor(t) for t in tops]
                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = [kernel_h, kernel_w]
                node.attributes["strides"] = [stride_h, stride_w]
                node.attributes["pads"] = [pad_top, pad_left, pad_bottom, pad_right]
                node.attributes["group"] = group
                self.graph.add_node(node)

            elif l_type == "Pooling":
                pool_type = params.get(0, 0)  # 0=max, 1=avg
                kernel_w = params.get(1, 1)
                kernel_h = params.get(11, kernel_w)
                stride_w = params.get(2, 1)
                stride_h = params.get(12, stride_w)
                pad_left = params.get(3, 0)
                pad_right = params.get(14, pad_left)
                pad_top = params.get(13, pad_left)
                pad_bottom = params.get(15, pad_top)

                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]

                op = "MaxPool" if pool_type == 0 else "AveragePool"
                node = Node(op, inputs=inputs, outputs=outputs, name=name)
                node.attributes["kernel_shape"] = [kernel_h, kernel_w]
                node.attributes["strides"] = [stride_h, stride_w]
                node.attributes["pads"] = [pad_top, pad_left, pad_bottom, pad_right]
                self.graph.add_node(node)

            elif l_type == "ReLU":
                inputs = [self.get_tensor(b) for b in bottoms]
                outputs = [self.get_tensor(t) for t in tops]
                node = Node("Relu", inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)

            elif l_type == "Split":
                inputs = [self.get_tensor(b) for b in bottoms]
                # Split in NCNN just copies input to multiple tops. ONNX uses Identity or we can just route them.
                # For safety, we use Identity nodes
                for top in tops:
                    t = self.get_tensor(top)
                    node = Node("Identity", inputs=inputs, outputs=[t], name=f"{name}_{top}")
                    self.graph.add_node(node)

        # Mark tops of last layer as graph outputs
        if layers:
            last_tops = layers[-1]["tops"]
            for t in last_tops:
                tensor = self.get_tensor(t)
                if tensor not in self.graph.outputs:
                    self.graph.outputs.append(tensor)

        return self.graph
