"""Darknet to ONNX IR Mapper."""

from typing import Any

import numpy as np
from onnx9000.core.ir import Constant, Graph, Node, Variable


class DarknetMapper:
    """Mapper to convert Darknet parsed layers and weights to ONNX IR."""

    def __init__(self, layers: list[dict[str, Any]], weights: np.ndarray):
        """Initialize the mapper.

        Args:
            layers (List[Dict[str, Any]]): Parsed layers from the .cfg file.
            weights (np.ndarray): Parsed weights from the .weights file.
        """
        self.layers = layers
        self.weights = weights
        self.weight_ptr = 0
        self.graph = Graph("DarknetModel")

    def _read_weights(self, num: int) -> np.ndarray:
        """Read `num` weights from the flat array.

        Args:
            num (int): Number of floats to read.

        Returns:
            np.ndarray: Sliced weight array.
        """
        res = self.weights[self.weight_ptr : self.weight_ptr + num]
        self.weight_ptr += num
        return res

    def map(self) -> Graph:
        """Map layers to an ONNX IR graph.

        Returns:
            Graph: The mapped ONNX9000 Core IR Graph.
        """
        if not self.layers:
            return self.graph

        net_info = self.layers[0]
        if net_info.get("type") == "net":
            h = int(net_info.get("height", 416))
            w = int(net_info.get("width", 416))
            c = int(net_info.get("channels", 3))
            self.layers = self.layers[1:]
        else:
            h, w, c = 416, 416, 3

        x = Variable("input", shape=(1, c, h, w))
        self.graph.add_tensor(x)
        self.graph.inputs.append(x)

        current = x
        outputs = []

        for i, layer in enumerate(self.layers):
            l_type = layer.get("type")
            if l_type == "convolutional":
                filters = int(layer.get("filters", 1))
                size = int(layer.get("size", 1))
                stride = int(layer.get("stride", 1))
                pad = int(layer.get("pad", 0))
                padding = size // 2 if pad else 0
                batch_normalize = int(layer.get("batch_normalize", 0))
                groups = int(layer.get("groups", 1))
                channels = (
                    current.shape[1] if hasattr(current, "shape") and len(current.shape) > 1 else c
                )
                if not isinstance(channels, int):
                    channels = 3  # fallback

                if batch_normalize:
                    bn_w = self._read_weights(filters)
                    bn_b = self._read_weights(filters)
                    bn_rm = self._read_weights(filters)
                    bn_rv = self._read_weights(filters)

                    conv_w = self._read_weights((channels // groups) * filters * size * size)
                    conv_w = conv_w.reshape((filters, channels // groups, size, size))

                    w_tensor = Constant(f"conv_{i}_w", values=conv_w.tobytes(), shape=conv_w.shape)
                    self.graph.add_tensor(w_tensor)

                    conv_out = Variable(f"conv_{i}_out")
                    self.graph.add_tensor(conv_out)

                    conv_node = Node("Conv", inputs=[current, w_tensor], outputs=[conv_out])
                    conv_node.attributes["kernel_shape"] = [size, size]
                    conv_node.attributes["strides"] = [stride, stride]
                    conv_node.attributes["pads"] = [padding, padding, padding, padding]
                    conv_node.attributes["group"] = groups
                    self.graph.add_node(conv_node)

                    bn_scale = Constant(f"bn_{i}_scale", values=bn_w.tobytes(), shape=bn_w.shape)
                    bn_B = Constant(f"bn_{i}_B", values=bn_b.tobytes(), shape=bn_b.shape)
                    bn_mean = Constant(f"bn_{i}_mean", values=bn_rm.tobytes(), shape=bn_rm.shape)
                    bn_var = Constant(f"bn_{i}_var", values=bn_rv.tobytes(), shape=bn_rv.shape)

                    for t in [bn_scale, bn_B, bn_mean, bn_var]:
                        self.graph.add_tensor(t)

                    bn_out = Variable(f"bn_{i}_out")
                    self.graph.add_tensor(bn_out)

                    bn_node = Node(
                        "BatchNormalization",
                        inputs=[conv_out, bn_scale, bn_B, bn_mean, bn_var],
                        outputs=[bn_out],
                    )
                    self.graph.add_node(bn_node)
                    current = bn_out

                else:
                    conv_b = self._read_weights(filters)
                    conv_w = self._read_weights((channels // groups) * filters * size * size)
                    conv_w = conv_w.reshape((filters, channels // groups, size, size))

                    w_tensor = Constant(f"conv_{i}_w", values=conv_w.tobytes(), shape=conv_w.shape)
                    b_tensor = Constant(f"conv_{i}_b", values=conv_b.tobytes(), shape=conv_b.shape)
                    self.graph.add_tensor(w_tensor)
                    self.graph.add_tensor(b_tensor)

                    conv_out = Variable(f"conv_{i}_out")
                    self.graph.add_tensor(conv_out)

                    conv_node = Node(
                        "Conv", inputs=[current, w_tensor, b_tensor], outputs=[conv_out]
                    )
                    conv_node.attributes["kernel_shape"] = [size, size]
                    conv_node.attributes["strides"] = [stride, stride]
                    conv_node.attributes["pads"] = [padding, padding, padding, padding]
                    conv_node.attributes["group"] = groups
                    self.graph.add_node(conv_node)
                    current = conv_out

                activation = layer.get("activation", "linear")
                if activation == "leaky":
                    act_out = Variable(f"act_{i}_out")
                    self.graph.add_tensor(act_out)
                    act_node = Node("LeakyRelu", inputs=[current], outputs=[act_out])
                    act_node.attributes["alpha"] = 0.1
                    self.graph.add_node(act_node)
                    current = act_out
                elif activation == "mish":
                    act_out = Variable(f"act_{i}_out")
                    self.graph.add_tensor(act_out)
                    act_node = Node("Mish", inputs=[current], outputs=[act_out])
                    self.graph.add_node(act_node)
                    current = act_out

            elif l_type == "route":
                layers_idx = [
                    int(x.strip()) for x in layer.get("layers", "").split(",") if x.strip()
                ]
                route_inputs = []
                for idx in layers_idx:
                    if idx < 0:
                        route_inputs.append(outputs[i + idx])
                    else:
                        route_inputs.append(outputs[idx])

                if len(route_inputs) == 1:
                    current = route_inputs[0]
                else:
                    concat_out = Variable(f"concat_{i}_out")
                    self.graph.add_tensor(concat_out)
                    concat_node = Node("Concat", inputs=route_inputs, outputs=[concat_out])
                    concat_node.attributes["axis"] = 1
                    self.graph.add_node(concat_node)
                    current = concat_out

            elif l_type == "yolo":
                # YOLO layers in Darknet usually just indicate an output
                # For basic mapping, we can mark this as a graph output.
                pass

            elif l_type == "shortcut":
                from_idx = int(layer.get("from", -1))
                if from_idx < 0:
                    from_layer = outputs[i + from_idx]
                else:
                    from_layer = outputs[from_idx]

                add_out = Variable(f"add_{i}_out")
                self.graph.add_tensor(add_out)
                add_node = Node("Add", inputs=[current, from_layer], outputs=[add_out])
                self.graph.add_node(add_node)
                current = add_out

            elif l_type == "maxpool":
                stride = int(layer.get("stride", 1))
                size = int(layer.get("size", 1))
                padding = int(layer.get("padding", size - 1))

                pool_out = Variable(f"maxpool_{i}_out")
                self.graph.add_tensor(pool_out)
                pool_node = Node("MaxPool", inputs=[current], outputs=[pool_out])
                pool_node.attributes["kernel_shape"] = [size, size]
                pool_node.attributes["strides"] = [stride, stride]
                # simplification for pads
                pool_node.attributes["pads"] = [0, 0, padding, padding]
                self.graph.add_node(pool_node)
                current = pool_out

            outputs.append(current)

        # Ensure graph has outputs
        if current not in self.graph.outputs:
            self.graph.outputs.append(current)

        return self.graph
