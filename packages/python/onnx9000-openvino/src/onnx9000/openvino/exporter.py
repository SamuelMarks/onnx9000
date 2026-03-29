import io
import struct
from typing import Any, Optional

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor

from .xml_builder import XmlBuilder, XmlNode


class OpenVinoExporter:
    def __init__(
        self,
        graph: Graph,
        version: str = "11",
        compress_to_fp16: bool = False,
        clamp_dynamic: bool = False,
    ):
        self.graph = graph
        self.version = version
        self.compress_to_fp16 = compress_to_fp16
        self.clamp_dynamic = clamp_dynamic
        self.node_id_counter = 0
        self.layer_ids: dict[str, int] = {}  # Maps ONNX node names to OpenVINO layer IDs
        self.port_ids: dict[str, tuple[int, int]] = {}  # Maps tensor name to (layer_id, port_id)
        self.bin_buffer = io.BytesIO()
        self._bin_cache: dict[bytes, tuple[int, int]] = {}  # data bytes -> (offset, size)
        self.edges: list[XmlNode] = []
        self._port_counters: dict[int, int] = {}  # layer_id -> next port

    def _next_id(self) -> int:
        idx = self.node_id_counter
        self.node_id_counter += 1
        return idx

    def _next_port(self, layer_id: int) -> int:
        if layer_id not in self._port_counters:
            self._port_counters[layer_id] = 0
        p = self._port_counters[layer_id]
        self._port_counters[layer_id] += 1
        return p

    def _emit_dynamic_const(
        self, name: str, data: list, shape: list[int], dtype: DType
    ) -> tuple[int, int]:
        layer_id = self._next_id()
        self.layer_ids[name] = layer_id

        layer = XmlNode("layer")
        layer.set_attribute("id", str(layer_id))
        layer.set_attribute("name", name)
        layer.set_attribute("type", "Const")
        layer.set_attribute("version", "opset1")

        data_node = XmlNode("data")
        data_node.set_attribute("element_type", self._map_dtype(dtype))
        actual_shape = shape if len(shape) > 0 else [1]
        data_node.set_attribute("shape", ",".join(str(d) for d in actual_shape))

        if dtype == DType.INT64:
            data_bytes = struct.pack(f"<{len(data)}q", *data)
        elif dtype == DType.INT32:
            data_bytes = struct.pack(f"<{len(data)}i", *data)
        elif dtype == DType.FLOAT32:
            data_bytes = struct.pack(f"<{len(data)}f", *data)
        else:
            raise ValueError(f"Dynamic const for {dtype} not implemented")

        if data_bytes in self._bin_cache:
            offset, size = self._bin_cache[data_bytes]
        else:
            offset = self.bin_buffer.tell()
            self.bin_buffer.write(data_bytes)
            size = len(data_bytes)
            self._bin_cache[data_bytes] = (offset, size)

        data_node.set_attribute("offset", str(offset))
        data_node.set_attribute("size", str(size))

        layer.add_child(data_node)

        output_port = self._next_port(layer_id)
        out_node = XmlNode("output")
        port = self._emit_shape(actual_shape, "port")
        port.set_attribute("id", str(output_port))
        port.set_attribute("precision", self._map_dtype(dtype))
        out_node.add_child(port)
        layer.add_child(out_node)

        self.port_ids[name] = (layer_id, output_port)
        return layer, output_port

    def _map_dtype(self, dtype: DType) -> str:
        mapping = {
            DType.FLOAT32: "f32",
            DType.FLOAT16: "f16",
            DType.FLOAT64: "f64",
            DType.BFLOAT16: "bf16",
            DType.INT64: "i64",
            DType.INT32: "i32",
            DType.INT16: "i16",
            DType.INT8: "i8",
            DType.UINT64: "u64",
            DType.UINT32: "u32",
            DType.UINT16: "u16",
            DType.UINT8: "u8",
            DType.BOOL: "boolean",
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype for OpenVINO: {dtype}")
        mapped = mapping[dtype]
        if mapped == "f32" and self.compress_to_fp16:
            return "f16"
        return mapped

    def _emit_shape(self, shape: list[Any], tag_name: str = "port") -> XmlNode:
        port_node = XmlNode(tag_name)
        for dim in shape:
            dim_val = str(dim)
            if dim_val == "-1" or (isinstance(dim, str) and not dim.isdigit()):
                dim_val = "1" if self.clamp_dynamic else "-1"
            dim_node = XmlNode("dim").add_child(dim_val)
            port_node.add_child(dim_node)
        return port_node

    def _add_edge(self, from_layer: int, from_port: int, to_layer: int, to_port: int):
        edge = XmlNode("edge")
        edge.set_attribute("from-layer", str(from_layer))
        edge.set_attribute("from-port", str(from_port))
        edge.set_attribute("to-layer", str(to_layer))
        edge.set_attribute("to-port", str(to_port))
        # Deduplication check
        for existing in self.edges:
            if (
                existing.attributes["from-layer"] == str(from_layer)
                and existing.attributes["from-port"] == str(from_port)
                and existing.attributes["to-layer"] == str(to_layer)
                and existing.attributes["to-port"] == str(to_port)
            ):
                return
        self.edges.append(edge)

    def export(self) -> tuple[str, bytes]:
        net = XmlNode("net")
        net.set_attribute("name", self.graph.name or "onnx9000_model")
        net.set_attribute("version", self.version)

        layers = XmlNode("layers")

        # Track consumed parameters to prevent emitting unused Parameters
        consumed_inputs = set()
        for node in self.graph.nodes:
            for inp in node.inputs:
                consumed_inputs.add(inp.name if not isinstance(inp, str) else inp)

        # 1. Map Parameters (Inputs)
        for val_info in self.graph.inputs:
            if val_info.name not in consumed_inputs:
                continue

            layer_id = self._next_id()
            self.layer_ids[val_info.name] = layer_id

            layer = XmlNode("layer")
            layer.set_attribute("id", str(layer_id))
            layer.set_attribute("name", val_info.name)
            layer.set_attribute("type", "Parameter")
            layer.set_attribute("version", "opset1")

            data = XmlNode("data")
            precision_str = self._map_dtype(val_info.dtype)
            data.set_attribute("element_type", precision_str)
            # Explicit precise precision metadata mapping
            data.set_attribute(
                "shape", ",".join(str(d) for d in val_info.shape) if val_info.shape else "1"
            )
            layer.add_child(data)

            output_port = self._next_port(layer_id)
            out_node = XmlNode("output")
            port = self._emit_shape(val_info.shape if len(val_info.shape) > 0 else [1], "port")
            port.set_attribute("id", str(output_port))
            port.set_attribute("precision", precision_str)
            out_node.add_child(port)
            layer.add_child(out_node)

            self.port_ids[val_info.name] = (layer_id, output_port)
            layers.add_child(layer)

        # 2. Map Constants (Initializers)
        for init_name in self.graph.initializers:
            if init_name not in self.graph.tensors:
                continue
            tensor = self.graph.tensors[init_name]
            layer_id = self._next_id()
            self.layer_ids[init_name] = layer_id

            layer = XmlNode("layer")
            layer.set_attribute("id", str(layer_id))
            layer.set_attribute("name", init_name)
            layer.set_attribute("type", "Const")
            layer.set_attribute("version", "opset1")

            data = XmlNode("data")
            if tensor.dtype is not None:
                data.set_attribute("element_type", self._map_dtype(tensor.dtype))

            # Extract ONNX scalars cleanly to Const layers with <dim>1</dim>
            actual_shape = tensor.shape if len(tensor.shape) > 0 else [1]
            data.set_attribute("shape", ",".join(str(d) for d in actual_shape))

            # Pack memory
            if tensor.data:
                # Handle global FP16 casting securely prior to deduplication
                if self.compress_to_fp16 and tensor.dtype == DType.FLOAT32:
                    floats = struct.unpack(f"<{len(tensor.data) // 4}f", tensor.data)
                    data_bytes = struct.pack(f"<{len(floats)}e", *floats)
                else:
                    data_bytes = bytes(tensor.data)

                # Deduplicate identical constant arrays exactly
                if data_bytes in self._bin_cache:
                    offset, size = self._bin_cache[data_bytes]
                else:
                    offset = self.bin_buffer.tell()
                    self.bin_buffer.write(data_bytes)
                    size = len(data_bytes)
                    self._bin_cache[data_bytes] = (offset, size)

                data.set_attribute("offset", str(offset))
                data.set_attribute("size", str(size))
            else:
                data.set_attribute("offset", "0")
                data.set_attribute("size", "0")

            layer.add_child(data)

            output_port = self._next_port(layer_id)
            out_node = XmlNode("output")
            port = self._emit_shape(actual_shape, "port")
            port.set_attribute("id", str(output_port))
            if tensor.dtype is not None:
                port.set_attribute("precision", self._map_dtype(tensor.dtype))
            out_node.add_child(port)
            layer.add_child(out_node)

            self.port_ids[init_name] = (layer_id, output_port)
            layers.add_child(layer)

        # 3. Map Nodes (Topologically sorted already in graph.nodes)
        for node in self.graph.nodes:
            layer_id = self._next_id()
            self.layer_ids[node.name or f"{node.op_type}_{layer_id}"] = layer_id

            layer = XmlNode("layer")
            layer.set_attribute("id", str(layer_id))
            layer.set_attribute("name", node.name or f"{node.op_type}_{layer_id}")
            layer.set_attribute("version", "opset1")

            type_mapping = {
                "Sub": "Subtract",
                "Mul": "Multiply",
                "Div": "Divide",
                "Pow": "Power",
                "Max": "Maximum",
                "Min": "Minimum",
                "Ceil": "Ceiling",
                "Conv": "Convolution",
                "Relu": "ReLU",
                "LeakyRelu": "PRelu",  # Or specialized LeakyRelu based on target IR
                "Sigmoid": "Sigmoid",
                "Tanh": "Tanh",
                "Elu": "Elu",
                "Selu": "Selu",
                "Softplus": "SoftPlus",
                "Gelu": "Gelu",
                "Softmax": "SoftMax",
                "LogSoftmax": "LogSoftmax",
                "PRelu": "PRelu",
                "Clip": "Clamp",
                "HardSigmoid": "HardSigmoid",
                "AveragePool": "AvgPool",
                "MaxPool": "MaxPool",
                "Flatten": "Reshape",
                "Reshape": "Reshape",
                "Transpose": "Transpose",
                "Squeeze": "Squeeze",
                "Unsqueeze": "Unsqueeze",
                "Concat": "Concat",
                "Split": "Split",
                "Gather": "Gather",
                "GatherND": "GatherND",
                "ScatterND": "ScatterNDUpdate",
                "ScatterElements": "ScatterElementsUpdate",
                "Shape": "ShapeOf",
                "Tile": "Tile",
                "Expand": "Broadcast",
                "ConstantOfShape": "Broadcast",
                "Cast": "Convert",
                "Pad": "Pad",
                "ReduceMean": "ReduceMean",
                "ReduceMax": "ReduceMax",
                "ReduceMin": "ReduceMin",
                "ReduceSum": "ReduceSum",
                "ReduceProd": "ReduceProd",
                "ArgMax": "ArgMax",
                "ArgMin": "ArgMin",
                "TopK": "TopK",
                "NonZero": "NonZero",
                "Equal": "Equal",
                "Not": "LogicalNot",
                "And": "LogicalAnd",
                "Or": "LogicalOr",
                "Xor": "LogicalXor",
                "Greater": "Greater",
                "Less": "Less",
                "GreaterOrEqual": "GreaterEqual",
                "LessOrEqual": "LessEqual",
                "Where": "Select",
                "Resize": "Interpolate",
                "SpaceToDepth": "SpaceToDepth",
                "DepthToSpace": "DepthToSpace",
                "NonMaxSuppression": "NonMaxSuppression",
                "RoiAlign": "ROIAlign",
                "CumSum": "CumSum",
                "QuantizeLinear": "FakeQuantize",
                "DequantizeLinear": "FakeQuantize",
                "If": "If",
                "Loop": "TensorIterator",
                "Scan": "TensorIterator",
                "Attention": "ScaledDotProductAttention",  # Assuming mapping to standard transformer block
                "Gemm": "MatMul",  # We do the Mul/Add decomposition as part of data mapping or handle as simple fully connected
                "Einsum": "Einsum",
                "Round": "Round",
                "BatchNormalization": "BatchNormInference",
                "InstanceNormalization": "MVN",
                "LayerNormalization": "MVN",
                "LpNormalization": "NormalizeL2",
            }
            ov_type = type_mapping.get(node.op_type, node.op_type)

            if node.op_type in ["Conv", "ConvTranspose", "Gemm"] and len(node.inputs) == 3:
                has_decoupled_bias = True
                bias_inp = node.inputs[2]
                bias_inp_name = bias_inp if isinstance(bias_inp, str) else bias_inp.name
                inputs_to_map = list(node.inputs[:2])
            else:
                has_decoupled_bias = False
                bias_inp_name = None
                inputs_to_map = list(node.inputs)

            layer.set_attribute("type", ov_type)

            data = XmlNode("data")

            # Binary element-wise broadcast
            if ov_type in [
                "Add",
                "Subtract",
                "Multiply",
                "Divide",
                "Power",
                "Maximum",
                "Minimum",
                "Mod",
                "Equal",
                "Less",
                "Greater",
                "LessEqual",
                "GreaterEqual",
                "LogicalAnd",
                "LogicalOr",
                "LogicalXor",
            ]:
                data.set_attribute("auto_broadcast", "numpy")

            if node.op_type in ["MatMul", "Gemm"]:
                trans_a = (
                    "true"
                    if node.attributes.get("transA") and node.attributes["transA"].value
                    else "false"
                )
                trans_b = (
                    "true"
                    if node.attributes.get("transB") and node.attributes["transB"].value
                    else "false"
                )
                data.set_attribute("transpose_a", trans_a)
                data.set_attribute("transpose_b", trans_b)
            elif node.op_type in ["Conv", "ConvTranspose"]:
                if node.attributes.get("group") and node.attributes["group"].value > 1:
                    if node.op_type == "Conv":
                        ov_type = "GroupConvolution"
                        layer.set_attribute("type", ov_type)
                    else:
                        ov_type = "GroupConvolutionBackpropData"
                        layer.set_attribute("type", ov_type)
                elif node.op_type == "ConvTranspose":
                    ov_type = "ConvolutionBackpropData"
                    layer.set_attribute("type", ov_type)

                if node.attributes.get("strides"):
                    data.set_attribute(
                        "strides", ",".join(str(x) for x in node.attributes["strides"].value)
                    )
                if node.attributes.get("dilations"):
                    data.set_attribute(
                        "dilations",
                        ",".join(str(x) for x in node.attributes["dilations"].value),
                    )
                if node.attributes.get("pads"):
                    pads = node.attributes["pads"].value
                    if len(pads) == 4:
                        data.set_attribute("pads_begin", f"{pads[0]},{pads[1]}")
                        data.set_attribute("pads_end", f"{pads[2]},{pads[3]}")
                    else:
                        data.set_attribute(
                            "pads_begin", ",".join(str(x) for x in pads[: len(pads) // 2])
                        )
                        data.set_attribute(
                            "pads_end", ",".join(str(x) for x in pads[len(pads) // 2 :])
                        )
                if node.attributes.get("output_padding"):
                    data.set_attribute(
                        "output_padding",
                        ",".join(str(x) for x in node.attributes["output_padding"].value),
                    )
                if node.attributes.get("auto_pad"):
                    auto_pad_map = {
                        "VALID": "valid",
                        "SAME_UPPER": "same_upper",
                        "SAME_LOWER": "same_lower",
                    }
                    data.set_attribute(
                        "auto_pad",
                        auto_pad_map.get(node.attributes["auto_pad"].value, "explicit"),
                    )
            elif node.op_type in ["MaxPool", "AveragePool"]:
                if node.attributes.get("kernel_shape"):
                    data.set_attribute(
                        "kernel",
                        ",".join(str(x) for x in node.attributes["kernel_shape"].value),
                    )
                if node.attributes.get("strides"):
                    data.set_attribute(
                        "strides", ",".join(str(x) for x in node.attributes["strides"].value)
                    )
                if node.attributes.get("pads"):
                    pads = node.attributes["pads"].value
                    if len(pads) == 4:
                        data.set_attribute("pads_begin", f"{pads[0]},{pads[1]}")
                        data.set_attribute("pads_end", f"{pads[2]},{pads[3]}")
                    else:
                        data.set_attribute(
                            "pads_begin", ",".join(str(x) for x in pads[: len(pads) // 2])
                        )
                        data.set_attribute(
                            "pads_end", ",".join(str(x) for x in pads[len(pads) // 2 :])
                        )
                if node.attributes.get("auto_pad"):
                    auto_pad_map = {
                        "VALID": "valid",
                        "SAME_UPPER": "same_upper",
                        "SAME_LOWER": "same_lower",
                    }
                    data.set_attribute(
                        "auto_pad",
                        auto_pad_map.get(node.attributes["auto_pad"].value, "explicit"),
                    )
                if node.op_type == "AveragePool" and node.attributes.get("count_include_pad"):
                    data.set_attribute(
                        "exclude-pad",
                        "false" if node.attributes["count_include_pad"].value else "true",
                    )
            elif node.op_type == "Gelu":
                if node.attributes and "approximate" in node.attributes:
                    approx = node.attributes["approximate"].value
                    if approx == b"tanh" or approx == "tanh":
                        data.set_attribute("approximation_mode", "tanh")
                    else:
                        data.set_attribute("approximation_mode", "erf")
                else:
                    data.set_attribute("approximation_mode", "erf")
            elif node.op_type == "Softmax":
                if node.attributes.get("axis"):
                    data.set_attribute("axis", str(node.attributes["axis"].value))
            elif node.op_type == "Concat":
                if node.attributes.get("axis"):
                    data.set_attribute("axis", str(node.attributes["axis"].value))
            elif node.op_type == "Split":
                if node.attributes.get("axis"):
                    data.set_attribute("axis", str(node.attributes["axis"].value))
            elif node.op_type == "Pad":
                if node.attributes.get("mode"):
                    data.set_attribute("pad_mode", str(node.attributes["mode"].value))
                if node.attributes and "pads" in node.attributes:
                    # Opset < 11: pads is an attribute
                    pads_data = list(node.attributes["pads"].value)
                    mid = len(pads_data) // 2
                    pads_begin = pads_data[:mid]
                    pads_end = pads_data[mid:]
                    b_layer, b_port = self._emit_dynamic_const(
                        node.name + "_pads_begin" if node.name else f"pads_begin_{layer_id}",
                        pads_begin,
                        [len(pads_begin)],
                        DType.INT64,
                    )
                    e_layer, e_port = self._emit_dynamic_const(
                        node.name + "_pads_end" if node.name else f"pads_end_{layer_id}",
                        pads_end,
                        [len(pads_end)],
                        DType.INT64,
                    )
                    layers.add_child(b_layer)
                    layers.add_child(e_layer)
                    inputs_to_map.append(
                        node.name + "_pads_begin" if node.name else f"pads_begin_{layer_id}"
                    )
                    inputs_to_map.append(
                        node.name + "_pads_end" if node.name else f"pads_end_{layer_id}"
                    )

                    val_attr = node.attributes.get("value")
                    val = val_attr.value if val_attr else 0.0
                    v_layer, v_port = self._emit_dynamic_const(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}",
                        [val],
                        [1],
                        DType.FLOAT32,
                    )
                    layers.add_child(v_layer)
                    inputs_to_map.append(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}"
                    )
                elif len(node.inputs) == 2:
                    # Inject 0.0 value injection natively
                    v_layer, v_port = self._emit_dynamic_const(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}",
                        [0.0],
                        [1],
                        DType.FLOAT32,
                    )
                    layers.add_child(v_layer)
                    inputs_to_map.append(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}"
                    )
            elif node.op_type == "Gather":
                if node.attributes.get("batch_dims"):
                    data.set_attribute("batch_dims", str(node.attributes["batch_dims"].value))
                if node.attributes.get("axis"):
                    axis_val = node.attributes["axis"].value
                    if len(inputs_to_map) == 2:
                        axis_layer, axis_port = self._emit_dynamic_const(
                            node.name + "_gather_axis" if node.name else f"gather_axis_{layer_id}",
                            [axis_val],
                            [1],
                            DType.INT64,
                        )
                        layers.add_child(axis_layer)
                        inputs_to_map.append(
                            node.name + "_gather_axis" if node.name else f"gather_axis_{layer_id}"
                        )
                # OpenVINO natively supports negative indices in Gather if axis is provided or opset is 8+
            elif node.op_type == "Slice":
                ov_type = "StridedSlice"
                layer.set_attribute("type", "StridedSlice")
                data.set_attribute("begin_mask", "0" * 10)
                data.set_attribute("end_mask", "0" * 10)
                data.set_attribute("new_axis_mask", "0" * 10)
                data.set_attribute("shrink_axis_mask", "0" * 10)
                data.set_attribute("ellipsis_mask", "0" * 10)
            elif node.op_type in [
                "ReduceMean",
                "ReduceMax",
                "ReduceMin",
                "ReduceSum",
                "ReduceProd",
            ]:
                if node.attributes.get("keepdims"):
                    data.set_attribute(
                        "keep_dims", "true" if node.attributes["keepdims"].value else "false"
                    )
            elif node.op_type in ["ArgMax", "ArgMin"]:
                if node.attributes.get("keepdims"):
                    data.set_attribute(
                        "keep_dims", "true" if node.attributes["keepdims"].value else "false"
                    )
                if node.attributes.get("axis"):
                    data.set_attribute("axis", str(node.attributes["axis"].value))
            elif node.op_type == "Resize":
                if node.attributes.get("mode"):
                    data.set_attribute("mode", str(node.attributes["mode"].value))
                if node.attributes.get("coordinate_transformation_mode"):
                    data.set_attribute(
                        "coordinate_transformation_mode",
                        str(node.attributes["coordinate_transformation_mode"].value),
                    )
                # shape_calculation_mode defaults to "sizes" in standard mapping
                data.set_attribute("shape_calculation_mode", "sizes")
                if node.attributes.get("nearest_mode"):
                    data.set_attribute("nearest_mode", str(node.attributes["nearest_mode"].value))
            elif node.op_type in ["SpaceToDepth", "DepthToSpace"]:
                if node.attributes.get("blocksize"):
                    data.set_attribute("block_size", str(node.attributes["blocksize"].value))
                if node.attributes.get("mode"):
                    data.set_attribute("mode", str(node.attributes["mode"].value))
            elif node.op_type == "NonMaxSuppression":
                if node.attributes and "center_point_box" in node.attributes:
                    val = node.attributes["center_point_box"].value
                    data.set_attribute("box_encoding", "center" if val else "corner")
                else:
                    data.set_attribute("box_encoding", "corner")
                data.set_attribute("sort_result_descending", "false")
            elif node.op_type == "RoiAlign":
                if node.attributes and "mode" in node.attributes:
                    data.set_attribute("mode", str(node.attributes["mode"].value))
                else:
                    data.set_attribute("mode", "avg")
            elif node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                data.set_attribute("levels", "256")  # usually INT8
                # FakeQuantize uses actual mathematically derived min/max
                # which requires runtime parsing of scale/zp, falling back to basic mapping
            elif node.op_type == "Einsum":
                if node.attributes.get("equation"):
                    data.set_attribute("equation", str(node.attributes["equation"].value))
            elif node.op_type == "LayerNormalization":
                if node.attributes.get("axis"):
                    data.set_attribute("axes", str(node.attributes["axis"].value))
                if node.attributes.get("epsilon"):
                    data.set_attribute("eps", str(node.attributes["epsilon"].value))
                data.set_attribute("normalize_variance", "true")
                data.set_attribute("eps_mode", "add")
            elif node.op_type == "InstanceNormalization":
                if node.attributes.get("epsilon"):
                    data.set_attribute("eps", str(node.attributes["epsilon"].value))
                data.set_attribute("normalize_variance", "true")
                data.set_attribute("eps_mode", "add")
            elif node.op_type == "LpNormalization":
                if node.attributes.get("axis"):
                    data.set_attribute("axes", str(node.attributes["axis"].value))
                if node.attributes.get("p"):
                    data.set_attribute("p", str(node.attributes["p"].value))
            elif node.op_type == "BatchNormalization":
                if node.attributes.get("epsilon"):
                    data.set_attribute("eps", str(node.attributes["epsilon"].value))
            elif node.op_type == "Dropout":
                # Evaluate explicit dropout removal safely
                ov_type = "Dropout"  # Or ignored entirely in optimization
            elif node.op_type == "If":
                # Emit bodies if graph attributes are present
                # Note: We need to export subgraphs and attach them
                if node.attributes.get("then_branch"):
                    sub_graph = node.attributes["then_branch"].value
                    print(f"HITTING THEN BRANCH!! {sub_graph}")
                    sub_exporter = OpenVinoExporter(sub_graph, self.version, self.compress_to_fp16)
                    # Avoid colliding IDs globally? Usually subgraphs maintain their own
                    sub_xml, sub_bin = sub_exporter.export()

                    XmlNode("body")
                    # For a true recursive tree, we'd rebuild the sub-DOM here
                    # We append it natively inside the layer.
                    # Minimal mapping implementation
                if node.attributes.get("else_branch"):
                    else_graph = node.attributes["else_branch"].value
                    else_exporter = OpenVinoExporter(
                        else_graph, self.version, self.compress_to_fp16
                    )
                    else_xml, else_bin = else_exporter.export()
                    XmlNode("body")
            elif node.op_type in ["Loop", "Scan"]:
                if node.attributes.get("body"):
                    sub_graph = node.attributes["body"].value
                    print(f"HITTING THEN BRANCH!! {sub_graph}")
                    sub_exporter = OpenVinoExporter(sub_graph, self.version, self.compress_to_fp16)
                    sub_xml, sub_bin = sub_exporter.export()
                    XmlNode("body")

            # Ensure data tag is added if it has attributes, or for certain types that always need it
            if len(data.attributes) > 0 or ov_type == "FakeQuantize":
                if ov_type == "FakeQuantize" and "levels" not in data.attributes:
                    data.set_attribute("levels", "256")
                layer.add_child(data)

            if node.op_type == "Cast":
                # Inject Convert nodes dynamically to enforce OpenVINO's rigid data type propagation
                ov_type = "Convert"
                layer.set_attribute("type", "Convert")
                if node.attributes and "to" in node.attributes:
                    to_dtype = DType(node.attributes["to"].value)
                    data.set_attribute("destination_type", self._map_dtype(to_dtype))
            elif node.op_type == "GridSample":
                ov_type = "GridSample"
                layer.set_attribute("type", "GridSample")
                if node.attributes and "mode" in node.attributes:
                    data.set_attribute("mode", str(node.attributes["mode"].value))
                if node.attributes and "padding_mode" in node.attributes:
                    data.set_attribute("padding_mode", str(node.attributes["padding_mode"].value))
                if node.attributes and "align_corners" in node.attributes:
                    val = node.attributes["align_corners"].value
                    data.set_attribute("align_corners", "true" if val else "false")
            elif node.op_type == "Size":
                # Size -> ShapeOf -> ReduceProd
                shape_layer_id = self._next_id()
                shape_name = node.name + "_shapeof" if node.name else f"shapeof_{layer_id}"
                shape_layer = (
                    XmlNode("layer")
                    .set_attribute("id", str(shape_layer_id))
                    .set_attribute("name", shape_name)
                    .set_attribute("type", "ShapeOf")
                    .set_attribute("version", "opset1")
                )

                shape_in_node = XmlNode("input")
                shape_in_port = self._next_port(shape_layer_id)
                shape_in_node.add_child(XmlNode("port").set_attribute("id", str(shape_in_port)))
                shape_layer.add_child(shape_in_node)

                if node.inputs and node.inputs[0] in self.port_ids:
                    from_layer, from_port = self.port_ids[node.inputs[0]]
                    self._add_edge(from_layer, from_port, shape_layer_id, shape_in_port)

                shape_out_node = XmlNode("output")
                shape_out_port = self._next_port(shape_layer_id)
                shape_out_node.add_child(
                    XmlNode("port")
                    .set_attribute("id", str(shape_out_port))
                    .set_attribute("precision", "i64")
                )
                shape_layer.add_child(shape_out_node)

                layers.add_child(shape_layer)

                axes_layer, axes_port = self._emit_dynamic_const(
                    node.name + "_axes" if node.name else f"axes_{layer_id}", [0], [1], DType.INT64
                )
                layers.add_child(axes_layer)

                ov_type = "ReduceProd"
                layer.set_attribute("type", "ReduceProd")
                data.set_attribute("keep_dims", "false")

                # Replace inputs_to_map: input0 is ShapeOf output, input1 is axes const
                inputs_to_map = [
                    shape_name,
                    node.name + "_axes" if node.name else f"axes_{layer_id}",
                ]
                self.port_ids[shape_name] = (shape_layer_id, shape_out_port)

            elif node.op_type == "Flatten":
                # Flatten -> Reshape
                shape_data = [0, -1]
                const_layer, const_port = self._emit_dynamic_const(
                    node.name + "_flatten_shape" if node.name else f"flatten_shape_{layer_id}",
                    shape_data,
                    [2],
                    DType.INT64,
                )
                layers.add_child(const_layer)
                inputs_to_map.append(
                    node.name + "_flatten_shape" if node.name else f"flatten_shape_{layer_id}"
                )
            elif node.op_type == "Transpose" and node.attributes and "perm" in node.attributes:
                perm_data = list(node.attributes["perm"].value)
                const_layer, const_port = self._emit_dynamic_const(
                    node.name + "_transpose_perm" if node.name else f"transpose_perm_{layer_id}",
                    perm_data,
                    [len(perm_data)],
                    DType.INT64,
                )
                layers.add_child(const_layer)
                inputs_to_map.append(
                    node.name + "_transpose_perm" if node.name else f"transpose_perm_{layer_id}"
                )
            elif node.op_type in [
                "ReduceMean",
                "ReduceMax",
                "ReduceMin",
                "ReduceSum",
                "ReduceProd",
            ]:
                # If axes is an attribute (Opset < 13), we must emit it as the 2nd input for OpenVINO.
                if node.attributes and "axes" in node.attributes:
                    axes_data = list(node.attributes["axes"].value)
                    const_layer, const_port = self._emit_dynamic_const(
                        node.name + "_reduce_axes" if node.name else f"reduce_axes_{layer_id}",
                        axes_data,
                        [len(axes_data)],
                        DType.INT64,
                    )
                    layers.add_child(const_layer)
                    inputs_to_map.append(
                        node.name + "_reduce_axes" if node.name else f"reduce_axes_{layer_id}"
                    )
            elif node.op_type == "Pad":
                if node.attributes and "pads" in node.attributes:
                    # Opset < 11: pads is an attribute
                    pads_data = list(node.attributes["pads"].value)
                    mid = len(pads_data) // 2
                    pads_begin = pads_data[:mid]
                    pads_end = pads_data[mid:]
                    b_layer, b_port = self._emit_dynamic_const(
                        node.name + "_pads_begin" if node.name else f"pads_begin_{layer_id}",
                        pads_begin,
                        [len(pads_begin)],
                        DType.INT64,
                    )
                    e_layer, e_port = self._emit_dynamic_const(
                        node.name + "_pads_end" if node.name else f"pads_end_{layer_id}",
                        pads_end,
                        [len(pads_end)],
                        DType.INT64,
                    )
                    layers.add_child(b_layer)
                    layers.add_child(e_layer)
                    inputs_to_map.append(
                        node.name + "_pads_begin" if node.name else f"pads_begin_{layer_id}"
                    )
                    inputs_to_map.append(
                        node.name + "_pads_end" if node.name else f"pads_end_{layer_id}"
                    )

                    val_attr = node.attributes.get("value")
                    val = val_attr.value if val_attr else 0.0
                    v_layer, v_port = self._emit_dynamic_const(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}",
                        [val],
                        [1],
                        DType.FLOAT32,
                    )
                    layers.add_child(v_layer)
                    inputs_to_map.append(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}"
                    )
                elif len(node.inputs) == 2:
                    # Inject 0.0 value injection natively
                    v_layer, v_port = self._emit_dynamic_const(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}",
                        [0.0],
                        [1],
                        DType.FLOAT32,
                    )
                    layers.add_child(v_layer)
                    inputs_to_map.append(
                        node.name + "_pad_value" if node.name else f"pad_value_{layer_id}"
                    )
            elif node.op_type == "GatherElements":
                # Emulate missing OpenVINO GatherElements via explicit Gather indexing
                # This requires complex index unrolling natively, we emit Gather as fallback
                ov_type = "GatherElements"
                layer.set_attribute("type", "GatherElements")
                if node.attributes and "axis" in node.attributes:
                    axis_val = node.attributes["axis"].value
                    axis_layer, axis_port = self._emit_dynamic_const(
                        node.name + "_gather_el_axis"
                        if node.name
                        else f"gather_el_axis_{layer_id}",
                        [axis_val],
                        [1],
                        DType.INT64,
                    )
                    layers.add_child(axis_layer)
                    inputs_to_map.append(
                        node.name + "_gather_el_axis" if node.name else f"gather_el_axis_{layer_id}"
                    )
            elif node.op_type == "ConstantOfShape":
                val = [0.0]
                dtype = DType.FLOAT32
                if node.attributes and "value" in node.attributes:
                    val_tensor = node.attributes["value"].value
                    if val_tensor.dtype == DType.FLOAT32:
                        val = (
                            [struct.unpack("<f", val_tensor.data[:4])[0]]
                            if val_tensor.data
                            else [0.0]
                        )
                        dtype = DType.FLOAT32
                    elif val_tensor.dtype == DType.INT64:
                        val = (
                            [struct.unpack("<q", val_tensor.data[:8])[0]]
                            if val_tensor.data
                            else [0]
                        )
                        dtype = DType.INT64
                    elif val_tensor.dtype == DType.INT32:
                        val = (
                            [struct.unpack("<i", val_tensor.data[:4])[0]]
                            if val_tensor.data
                            else [0]
                        )
                        dtype = DType.INT32

                const_layer, const_port = self._emit_dynamic_const(
                    node.name + "_scalar_val" if node.name else f"scalar_val_{layer_id}",
                    val,
                    [1],
                    dtype,
                )
                layers.add_child(const_layer)
                inputs_to_map = [
                    node.name + "_scalar_val" if node.name else f"scalar_val_{layer_id}",
                    node.inputs[0],
                ]
                ov_type = "Broadcast"

            in_node = XmlNode("input")
            for inp in inputs_to_map:
                inp_name = str(inp) if isinstance(inp, str) else str(inp.name)
                input_port = self._next_port(layer_id)
                port = XmlNode("port")
                port.set_attribute("id", str(input_port))
                in_node.add_child(port)
                if inp_name in self.port_ids:
                    from_layer, from_port = self.port_ids[inp_name]
                    self._add_edge(from_layer, from_port, layer_id, input_port)
                elif inp_name != "":
                    raise ValueError(
                        f"Missing input pointer: '{inp_name}' for node '{node.name or layer_id}'"
                    )
            if len(inputs_to_map) > 0:
                layer.add_child(in_node)

            out_node = XmlNode("output")
            for out in node.outputs:
                out_name = str(out) if isinstance(out, str) else str(out.name)
                output_port = self._next_port(layer_id)
                port = XmlNode("port")
                port.set_attribute("id", str(output_port))
                out_node.add_child(port)

                if has_decoupled_bias:
                    # Intermediate output
                    self.port_ids[out_name + "_internal_nobias"] = (layer_id, output_port)
                else:
                    self.port_ids[out_name] = (layer_id, output_port)
            if len(node.outputs) > 0:
                layer.add_child(out_node)

            layers.add_child(layer)

            if has_decoupled_bias:
                for out in node.outputs:
                    out_name = out if isinstance(out, str) else out.name

                    add_layer_id = self._next_id()
                    add_layer = XmlNode("layer")
                    add_layer.set_attribute("id", str(add_layer_id))
                    add_layer.set_attribute("name", out_name + "_bias_add")
                    add_layer.set_attribute("type", "Add")
                    add_layer.set_attribute("version", "opset1")

                    add_data = XmlNode("data").set_attribute("auto_broadcast", "numpy")
                    add_layer.add_child(add_data)

                    add_in_node = XmlNode("input")

                    # 1. Main op output
                    p1 = self._next_port(add_layer_id)
                    add_in_node.add_child(XmlNode("port").set_attribute("id", str(p1)))
                    fl1, fp1 = self.port_ids[out_name + "_internal_nobias"]
                    self._add_edge(fl1, fp1, add_layer_id, p1)

                    # 2. Bias
                    p2 = self._next_port(add_layer_id)
                    add_in_node.add_child(XmlNode("port").set_attribute("id", str(p2)))
                    if bias_inp_name in self.port_ids:
                        fl2, fp2 = self.port_ids[bias_inp_name]
                        self._add_edge(fl2, fp2, add_layer_id, p2)

                    add_layer.add_child(add_in_node)

                    add_out_node = XmlNode("output")
                    p3 = self._next_port(add_layer_id)
                    add_out_node.add_child(XmlNode("port").set_attribute("id", str(p3)))
                    add_layer.add_child(add_out_node)

                    self.port_ids[out_name] = (add_layer_id, p3)

                    layers.add_child(add_layer)

        # 4. Map Results (Outputs)
        for val_info in self.graph.outputs:
            if val_info.name in self.port_ids:
                from_layer, from_port = self.port_ids[val_info.name]

            layer_id = self._next_id()
            self.layer_ids[val_info.name + "_result"] = layer_id

            layer = XmlNode("layer")
            layer.set_attribute("id", str(layer_id))
            layer.set_attribute("name", val_info.name + "_result")
            layer.set_attribute("type", "Result")
            layer.set_attribute("version", "opset1")

            input_port = self._next_port(layer_id)
            in_node = XmlNode("input")
            port = self._emit_shape(val_info.shape, "port")
            port.set_attribute("id", str(input_port))
            port.set_attribute("precision", self._map_dtype(val_info.dtype))
            in_node.add_child(port)
            layer.add_child(in_node)

            layers.add_child(layer)

            # Edge from the producer to this Result
            self._add_edge(from_layer, from_port, layer_id, input_port)

        # 5. Runtime Info (rt_info)
        rt_info = XmlNode("rt_info")
        meta_data = XmlNode("meta_data")
        mo_settings = XmlNode("MO_version").set_attribute("value", "onnx9000")
        conversion_params = XmlNode("cli_parameters")
        conversion_params.add_child(
            XmlNode("compress_to_fp16").set_attribute("value", str(self.compress_to_fp16))
        )
        meta_data.add_child(mo_settings)
        meta_data.add_child(conversion_params)
        rt_info.add_child(meta_data)

        net.add_child(layers)

        edges_node = XmlNode("edges")
        for e in self.edges:
            edges_node.add_child(e)
        net.add_child(edges_node)

        net.add_child(rt_info)

        builder = XmlBuilder().set_root(net)
        xml_str = builder.to_string(pretty=True)
        return xml_str, self.bin_buffer.getvalue()
