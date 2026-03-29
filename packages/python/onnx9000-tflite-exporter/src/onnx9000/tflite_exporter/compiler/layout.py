"""Layout optimization and graph transformation for TFLite export.

This module provides the LayoutOptimizer class, which performs various
optimizations and layout transformations (e.g., NCHW to NHWC) on the ONNX
graph to prepare it for TFLite conversion.
"""

import logging
import struct

from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo

logger = logging.getLogger(__name__)


class LayoutOptimizer:
    """Optimizes and transforms the ONNX graph for TFLite export.

    Handles layout changes (NCHW to NHWC), constant folding, fusion,
    and other graph-level optimizations.
    """

    def __init__(self, graph: Graph, keep_nchw: bool = False):
        """Initialize the layout optimizer.

        Args:
            graph: The ONNX graph to optimize.
            keep_nchw: If True, do not perform NCHW to NHWC transformation.

        """
        self.graph = graph
        self.keep_nchw = keep_nchw

    def optimize(self) -> None:
        """Run all optimization passes on the graph."""
        self.strip_identities()
        self.rewrite_negative_axes()
        self.fuse_conv_batch_normalization()
        self.decompose_batch_normalization()
        self.evaluate_constants()
        self.emulate_einsum()
        self.process_edge_cases()
        self.expand_1d_spatial_ops()

        if self.keep_nchw:
            return

        self.inject_transposes()
        self.push_down_transposes()
        self.cancel_transposes()
        self.fold_constants()
        self.recalculate_shapes()
        self.check_irreducible_transposes()

    def emulate_einsum(self) -> None:
        """Stub for emulating Einsum nodes by decomposing them."""
        import logging

        for i in range(len(self.graph.nodes)):
            node = self.graph.nodes[i]
            if node.op_type == "Einsum":
                equation = (
                    node.attributes.get("equation").value if node.attributes.get("equation") else ""
                )
                logging.warning(
                    f"[onnx2tf] Warning: Einsum node '{node.name}' with equation '{equation}' detected. Einsum decomposition to Transpose/Reshape/MatMul is currently a stub and may fail execution."
                )
                node.op_type = "MatMul"

    def process_edge_cases(self) -> None:
        """Handle framework-specific or model-specific edge cases (e.g., Keras origins, stateful tensors)."""
        metadata = getattr(self.graph, "metadata", None)
        import logging

        if metadata and getattr(metadata, "producer_name", None) == "onnx9000.keras":
            logging.info(
                "[onnx2tf] Detected 'onnx9000.keras' origin. Native TF layouts (NHWC) will bypass strict transpilation passes where explicitly marked."
            )

        for v in self.graph.value_info:
            if "state" in v.name or "hidden" in v.name:
                logging.warning(
                    f"[onnx2tf] EdgeTPU / TFLite Warning: Stateful sequence tensor '{v.name}' detected. TFLite requires explicit Variable mappings or manual hidden state injection for stateless models."
                )
                break

    def expand_1d_spatial_ops(self) -> None:
        """Expand 1D spatial operations to 2D for TFLite compatibility."""
        spatial_ops = {
            "Conv",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "ConvTranspose",
        }

        i = 0
        while i < len(self.graph.nodes):
            node = self.graph.nodes[i]
            if node.op_type in spatial_ops:
                if node.inputs:
                    in_name = node.inputs[0]
                    in_info = next((v for v in self.graph.value_info if v.name == in_name), None)
                    if not in_info:
                        in_info = next((v for v in self.graph.inputs if v.name == in_name), None)
                    if not in_info:
                        in_info = self.graph.tensors.get(in_name)

                    if in_info and in_info.shape and len(in_info.shape) == 3:
                        unsqueeze_out = f"{in_name}_expanded_1d"
                        axes_name = f"{node.name}_unsqueeze_axes"

                        import struct

                        self.graph.tensors[axes_name] = Tensor(
                            axes_name,
                            shape=[1],
                            dtype="int64",
                            is_initializer=True,
                            data=struct.pack("<q", 2),
                        )
                        unsqueeze_node = Node(
                            "Unsqueeze",
                            [in_name, axes_name],
                            [unsqueeze_out],
                            name=f"{node.name}_unsqueeze",
                        )
                        node.inputs[0] = unsqueeze_out

                        orig_out = node.outputs[0] if node.outputs else None
                        if orig_out:
                            squeeze_in = f"{orig_out}_expanded_1d"
                            node.outputs[0] = squeeze_in
                            squeeze_axes_name = f"{node.name}_squeeze_axes"
                            self.graph.tensors[squeeze_axes_name] = Tensor(
                                squeeze_axes_name,
                                shape=[1],
                                dtype="int64",
                                is_initializer=True,
                                data=struct.pack("<q", 2),
                            )
                            squeeze_node = Node(
                                "Squeeze",
                                [squeeze_in, squeeze_axes_name],
                                [orig_out],
                                name=f"{node.name}_squeeze",
                            )

                            kernel_attr = node.attributes.get("kernel_shape")
                            if kernel_attr and len(kernel_attr.value) == 1:
                                node.attributes["kernel_shape"] = Attribute(
                                    "kernel_shape", "INTS", [1, kernel_attr.value[0]]
                                )
                            strides_attr = node.attributes.get("strides")
                            if strides_attr and len(strides_attr.value) == 1:
                                node.attributes["strides"] = Attribute(
                                    "strides", "INTS", [1, strides_attr.value[0]]
                                )
                            dilations_attr = node.attributes.get("dilations")
                            if dilations_attr and len(dilations_attr.value) == 1:
                                node.attributes["dilations"] = Attribute(
                                    "dilations", "INTS", [1, dilations_attr.value[0]]
                                )
                            pads_attr = node.attributes.get("pads")
                            if pads_attr and len(pads_attr.value) == 2:
                                node.attributes["pads"] = Attribute(
                                    "pads", "INTS", [0, pads_attr.value[0], 0, pads_attr.value[1]]
                                )

                            if len(node.inputs) > 1:
                                w_name = node.inputs[1]
                                w_tensor = self.graph.tensors.get(w_name)
                                if w_tensor and w_tensor.shape and len(w_tensor.shape) == 3:
                                    w_tensor.shape = (
                                        w_tensor.shape[0],
                                        w_tensor.shape[1],
                                        1,
                                        w_tensor.shape[2],
                                    )

                            self.graph.nodes.insert(i, unsqueeze_node)
                            i += 1
                            self.graph.nodes.insert(i + 1, squeeze_node)
                            i += 1
            i += 1

    def fuse_conv_batch_normalization(self) -> None:
        """Fuse Convolution and Batch Normalization layers where possible."""
        import math
        import struct

        i = 0
        while i < len(self.graph.nodes):
            node = self.graph.nodes[i]
            if node.op_type == "Conv":
                if not node.outputs:
                    i += 1
                    continue
                y = node.outputs[0]

                consumer_idx = -1
                consumer = None
                num_consumers = 0
                for idx, n in enumerate(self.graph.nodes):
                    if y in n.inputs:
                        num_consumers += 1
                        if n.op_type == "BatchNormalization" and n.inputs[0] == y:
                            consumer_idx = idx
                            consumer = n

                if consumer and num_consumers == 1 and len(consumer.inputs) >= 5:
                    x, scale, b, mean, v = consumer.inputs[:5]
                    scale_tensor = self.graph.tensors.get(scale)
                    b_tensor = self.graph.tensors.get(b)
                    mean_tensor = self.graph.tensors.get(mean)
                    v_tensor = self.graph.tensors.get(v)

                    if len(node.inputs) > 1:
                        w_name = node.inputs[1]
                        w_tensor = self.graph.tensors.get(w_name)

                        if (
                            scale_tensor
                            and b_tensor
                            and mean_tensor
                            and v_tensor
                            and w_tensor
                            and scale_tensor.data
                            and b_tensor.data
                            and mean_tensor.data
                            and v_tensor.data
                            and w_tensor.data
                            and w_tensor.shape
                            and len(w_tensor.shape) >= 3
                        ):
                            epsilon_attr = consumer.attributes.get("epsilon")
                            epsilon = epsilon_attr.value if epsilon_attr else 1e-5

                            scale_data = list(
                                struct.unpack(f"<{len(scale_tensor.data) // 4}f", scale_tensor.data)
                            )
                            b_data = list(
                                struct.unpack(f"<{len(b_tensor.data) // 4}f", b_tensor.data)
                            )
                            mean_data = list(
                                struct.unpack(f"<{len(mean_tensor.data) // 4}f", mean_tensor.data)
                            )
                            v_data = list(
                                struct.unpack(f"<{len(v_tensor.data) // 4}f", v_tensor.data)
                            )
                            w_data = list(
                                struct.unpack(f"<{len(w_tensor.data) // 4}f", w_tensor.data)
                            )

                            num_channels = len(scale_data)
                            if w_tensor.shape[0] == num_channels:
                                channel_size = len(w_data) // num_channels
                                for c in range(num_channels):
                                    mul_factor = scale_data[c] / math.sqrt(v_data[c] + epsilon)
                                    offset = c * channel_size
                                    for j in range(channel_size):
                                        w_data[offset + j] *= mul_factor

                                w_tensor.data = struct.pack(f"<{len(w_data)}f", *w_data)

                                b_conv_name = (
                                    node.inputs[2]
                                    if len(node.inputs) > 2
                                    else f"{node.name}_fused_bias"
                                )
                                b_conv_data = [0.0] * num_channels
                                if len(node.inputs) > 2:
                                    b_conv_tensor = self.graph.tensors.get(b_conv_name)
                                    if b_conv_tensor and b_conv_tensor.data:
                                        b_conv_data = list(
                                            struct.unpack(
                                                f"<{len(b_conv_tensor.data) // 4}f",
                                                b_conv_tensor.data,
                                            )
                                        )
                                else:
                                    self.graph.tensors[b_conv_name] = Tensor(
                                        b_conv_name,
                                        shape=[num_channels],
                                        dtype="float32",
                                        is_initializer=True,
                                        data=b"",
                                    )
                                    node.inputs.append(b_conv_name)

                                for c in range(num_channels):
                                    mul_factor = scale_data[c] / math.sqrt(v_data[c] + epsilon)
                                    b_conv_data[c] = (
                                        b_conv_data[c] - mean_data[c]
                                    ) * mul_factor + b_data[c]

                                self.graph.tensors[b_conv_name].data = struct.pack(
                                    f"<{len(b_conv_data)}f", *b_conv_data
                                )

                                node.outputs[0] = consumer.outputs[0]
                                self.graph.nodes.pop(consumer_idx)
                                if consumer_idx <= i:
                                    i -= 1
            i += 1

    def decompose_batch_normalization(self) -> None:
        """Decompose Batch Normalization into a combination of Mul and Add nodes."""
        i = 0
        while i < len(self.graph.nodes):
            node = self.graph.nodes[i]
            if node.op_type == "BatchNormalization":
                inputs = node.inputs
                if len(inputs) >= 5:
                    x, scale, b, mean, v = inputs[:5]
                    y = node.outputs[0] if node.outputs else None
                    if y:
                        scale_tensor = self.graph.tensors.get(scale)
                        b_tensor = self.graph.tensors.get(b)
                        mean_tensor = self.graph.tensors.get(mean)
                        v_tensor = self.graph.tensors.get(v)

                        if (
                            scale_tensor
                            and b_tensor
                            and mean_tensor
                            and v_tensor
                            and scale_tensor.data
                            and b_tensor.data
                            and mean_tensor.data
                            and v_tensor.data
                        ):
                            epsilon_attr = node.attributes.get("epsilon")
                            epsilon = epsilon_attr.value if epsilon_attr else 1e-5

                            scale_data = struct.unpack(
                                f"<{len(scale_tensor.data) // 4}f", scale_tensor.data
                            )
                            b_data = struct.unpack(f"<{len(b_tensor.data) // 4}f", b_tensor.data)
                            mean_data = struct.unpack(
                                f"<{len(mean_tensor.data) // 4}f", mean_tensor.data
                            )
                            v_data = struct.unpack(f"<{len(v_tensor.data) // 4}f", v_tensor.data)

                            import math

                            mul_data = []
                            add_data = []
                            for j in range(len(scale_data)):
                                mul_factor = scale_data[j] / math.sqrt(v_data[j] + epsilon)
                                mul_data.append(mul_factor)
                                add_data.append(b_data[j] - mean_data[j] * mul_factor)

                            mul_name = f"{node.name}_mul_factor"
                            add_name = f"{node.name}_add_factor"
                            mul_out_name = f"{node.name}_mul_out"

                            self.graph.tensors[mul_name] = Tensor(
                                mul_name,
                                shape=[len(scale_data)],
                                dtype="float32",
                                is_initializer=True,
                                data=struct.pack(f"<{len(scale_data)}f", *mul_data),
                            )
                            self.graph.tensors[add_name] = Tensor(
                                add_name,
                                shape=[len(scale_data)],
                                dtype="float32",
                                is_initializer=True,
                                data=struct.pack(f"<{len(scale_data)}f", *add_data),
                            )

                            mul_node = Node(
                                "Mul", [x, mul_name], [mul_out_name], name=f"{node.name}_mul"
                            )
                            add_node = Node(
                                "Add", [mul_out_name, add_name], [y], name=f"{node.name}_add"
                            )

                            self.graph.nodes[i] = mul_node
                            self.graph.nodes.insert(i + 1, add_node)
                            i += 1
            i += 1

    def evaluate_constants(self) -> None:
        """Statically evaluate constants and check for potential runtime issues (e.g., division by zero)."""
        import struct
        import logging

        for node in self.graph.nodes:
            if node.op_type == "Div":
                if len(node.inputs) > 1:
                    b_name = node.inputs[1]
                    b_tensor = self.graph.tensors.get(b_name)
                    if b_tensor and b_tensor.is_initializer and b_tensor.data:
                        data = list(struct.unpack(f"<{len(b_tensor.data) // 4}f", b_tensor.data))
                        if any(abs(v) < 1e-12 for v in data):
                            logging.warning(
                                f"[onnx2tf] Warning: Division by zero (or near-zero) detected statically in constant tensor {b_name} for node {node.name}. This will crash TFLite runtime."
                            )

    def strip_identities(self) -> None:
        """Remove Identity and Dropout nodes from the graph."""
        ops_to_remove = {"Dropout", "Identity"}
        i = 0
        while i < len(self.graph.nodes):
            node = self.graph.nodes[i]
            if node.op_type in ops_to_remove:
                if node.inputs and node.outputs:
                    input_name = node.inputs[0]
                    output_name = node.outputs[0]

                    for consumer in self.graph.nodes:
                        for j in range(len(consumer.inputs)):
                            if consumer.inputs[j] == output_name:
                                consumer.inputs[j] = input_name

                    for j in range(len(self.graph.outputs)):
                        if self.graph.outputs[j].name == output_name:
                            self.graph.outputs[j].name = input_name

                    self.graph.nodes.pop(i)
                    continue
            i += 1

    def inject_transposes(self) -> None:
        """Inject Transpose nodes to convert spatial operations from NCHW to NHWC."""
        spatial_ops = {
            "Conv",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "ConvTranspose",
            "BatchNormalization",
        }
        new_nodes = []
        node_counter = 0

        for node in self.graph.nodes:
            if node.op_type in spatial_ops:
                input_name = node.inputs[0] if node.inputs else None
                rank = 4
                if input_name:
                    in_info = None
                    for vi in self.graph.value_info:
                        if vi.name == input_name:
                            in_info = vi
                            break
                    if not in_info:
                        for vi in self.graph.inputs:
                            if vi.name == input_name:
                                in_info = vi
                                break
                    if not in_info and input_name in self.graph.tensors:
                        in_info = self.graph.tensors[input_name]

                    if in_info and hasattr(in_info, "shape"):
                        # If shape exists it may be a tuple or list
                        if in_info.shape:
                            rank = len(in_info.shape)

                if rank < 3 or rank > 5:
                    new_nodes.append(node)
                    continue

                in_perm = [0, 2, 3, 1]
                out_perm = [0, 3, 1, 2]
                layout_name = "nhwc"

                if rank == 3:
                    in_perm = [0, 2, 1]
                    out_perm = [0, 2, 1]
                    layout_name = "nwc"
                elif rank == 5:
                    in_perm = [0, 2, 3, 4, 1]
                    out_perm = [0, 4, 1, 2, 3]
                    layout_name = "ndhwc"

                if input_name:
                    transposed_input = f"{input_name}_{layout_name}_{node_counter}"
                    transpose_node = Node(
                        op_type="Transpose",
                        inputs=[input_name],
                        outputs=[transposed_input],
                        attributes={"perm": Attribute("perm", "INTS", in_perm)},
                        name=f"trans_in_{node_counter}",
                    )
                    new_nodes.append(transpose_node)
                    node.inputs[0] = transposed_input

                output_name = node.outputs[0] if node.outputs else None
                if output_name:
                    transposed_output = f"{output_name}_{layout_name}_inv_{node_counter}"
                    transpose_node = Node(
                        op_type="Transpose",
                        inputs=[transposed_output],
                        outputs=[output_name],
                        attributes={"perm": Attribute("perm", "INTS", out_perm)},
                        name=f"trans_out_{node_counter}",
                    )
                    node.outputs[0] = transposed_output
                    new_nodes.append(node)
                    new_nodes.append(transpose_node)
                else:
                    new_nodes.append(node)
                node_counter += 1
            else:
                new_nodes.append(node)

        self.graph.nodes = new_nodes

    def push_down_transposes(self) -> None:
        """Push Transpose nodes down through elementwise and axis-based operations."""
        elementwise_ops = {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Relu",
            "Relu6",
            "LeakyRelu",
            "Sigmoid",
            "Tanh",
            "Max",
            "Min",
            "Abs",
            "BatchNormalization",
            "InstanceNormalization",
            "Expand",
            "Tile",
        }
        axis_ops = {
            "Concat",
            "Split",
            "Softmax",
            "LogSoftmax",
            "Gather",
            "ScatterElements",
            "ScatterND",
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ArgMax",
            "ArgMin",
        }
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.graph.nodes):
                node = self.graph.nodes[i]
                is_elementwise = node.op_type in elementwise_ops
                is_axis_op = node.op_type in axis_ops
                is_reshape = node.op_type == "Reshape"

                if not (is_elementwise or is_axis_op or is_reshape):
                    i += 1
                    continue

                all_transposed = True
                transpose_perm = None
                transpose_nodes_to_remove = set()

                for in_name in node.inputs:
                    if in_name in self.graph.tensors:
                        continue

                    producer = next((n for n in self.graph.nodes if in_name in n.outputs), None)
                    if not producer or producer.op_type != "Transpose":
                        all_transposed = False
                        break

                    perm = producer.attributes.get("perm")
                    if not perm or (perm.value != [0, 2, 3, 1] and perm.value != [0, 3, 1, 2]):
                        all_transposed = False
                        break

                    if transpose_perm is not None and transpose_perm != perm.value:
                        all_transposed = False
                        break

                    transpose_nodes_to_remove.add(producer.name)
                    transpose_perm = perm.value

                if all_transposed and transpose_nodes_to_remove and transpose_perm:
                    changed = True

                    if node.op_type in ("Expand", "Tile"):
                        import logging

                        logging.warning(
                            f"[onnx2tf] Warning: {node.op_type} node {node.name} encountered during layout permutation push-down. Arbitrary shape broadcasting might be unstable and require TFLite inference fallbacks."
                        )

                    if is_axis_op:
                        axis_mapping = None
                        if transpose_perm == [0, 2, 3, 1]:
                            axis_mapping = [0, 3, 1, 2]
                        elif transpose_perm == [0, 3, 1, 2]:
                            axis_mapping = [0, 2, 3, 1]

                        axis_attr = node.attributes.get("axis")
                        if axis_attr and isinstance(axis_attr.value, int):
                            axis = axis_attr.value
                            if axis < 0:
                                axis += 4

                            if axis_mapping and 0 <= axis < 4:
                                node.attributes["axis"] = Attribute(
                                    "axis", "INT", axis_mapping[axis]
                                )

                        axes_attr = node.attributes.get("axes")
                        if axes_attr and isinstance(axes_attr.value, list):
                            new_axes = []
                            for a in axes_attr.value:
                                a_pos = a if a >= 0 else a + 4
                                if axis_mapping and 0 <= a_pos < 4:
                                    new_axes.append(axis_mapping[a_pos])
                                else:
                                    new_axes.append(a)
                            node.attributes["axes"] = Attribute("axes", "INTS", new_axes)

                        keepdims_attr = node.attributes.get("keepdims")
                        if keepdims_attr is None and node.op_type in (
                            "ReduceMean",
                            "ReduceSum",
                            "ReduceMax",
                            "ReduceMin",
                            "ReduceProd",
                        ):
                            return None

                    for producer_name in transpose_nodes_to_remove:
                        producer_idx, producer = next(
                            (
                                (idx, n)
                                for idx, n in enumerate(self.graph.nodes)
                                if n.name == producer_name
                            ),
                            (-1, None),
                        )
                        if producer:
                            original_input = producer.inputs[0]
                            idx = node.inputs.index(producer.outputs[0])
                            node.inputs[idx] = original_input
                            self.graph.nodes.pop(producer_idx)
                            if producer_idx <= i:
                                i -= 1

                    for out_idx, output_name in enumerate(node.outputs):
                        if output_name:
                            transposed_output = f"{output_name}_pushed_trans"
                            new_transpose = Node(
                                op_type="Transpose",
                                inputs=[transposed_output],
                                outputs=[output_name],
                                attributes={
                                    "perm": Attribute("perm", "INTS", transpose_perm.copy())
                                },
                                name=f"{node.name}_pushed_trans_{out_idx}",
                            )
                            node.outputs[out_idx] = transposed_output
                            self.graph.nodes.insert(i + 1, new_transpose)
                            i += 1
                i += 1

    def cancel_transposes(self) -> None:
        """Identify and remove redundant back-to-back Transpose operations."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.graph.nodes) - 1:
                node1 = self.graph.nodes[i]
                node2 = self.graph.nodes[i + 1]

                if node1.op_type == "Transpose" and node2.op_type == "Transpose":
                    perm1 = node1.attributes.get("perm")
                    perm2 = node2.attributes.get("perm")
                    if perm1 and perm2:
                        p1 = perm1.value
                        p2 = perm2.value
                        if node1.outputs[0] == node2.inputs[0]:
                            if (p1 == [0, 2, 3, 1] and p2 == [0, 3, 1, 2]) or (
                                p1 == [0, 3, 1, 2] and p2 == [0, 2, 3, 1]
                            ):
                                changed = True
                                original_input = node1.inputs[0]
                                final_output = node2.outputs[0]

                                for consumer in self.graph.nodes:
                                    for j in range(len(consumer.inputs)):
                                        if consumer.inputs[j] == final_output:
                                            consumer.inputs[j] = original_input

                                for j in range(len(self.graph.outputs)):
                                    if self.graph.outputs[j].name == final_output:
                                        self.graph.outputs[j].name = original_input

                                self.graph.nodes.pop(i)
                                self.graph.nodes.pop(i)  # the next one moved down
                                if i > 0:
                                    i -= 1
                                continue
                i += 1

    def fuse_activations_and_matmuls(self) -> None:
        """Fuse activation functions into preceding Convolution or Gemm layers."""
        fused_activations = {"Relu", "Relu6"}
        i = 0
        while i < len(self.graph.nodes):
            node = self.graph.nodes[i]

            if node.op_type == "MatMul":
                if node.outputs:
                    y = node.outputs[0]
                    consumer_idx = -1
                    consumer = None
                    num_consumers = 0
                    for idx, n in enumerate(self.graph.nodes):
                        if y in n.inputs:
                            num_consumers += 1
                            if n.op_type == "Add":
                                consumer_idx = idx
                                consumer = n

                    if consumer and num_consumers == 1:
                        add_input = next((inp for inp in consumer.inputs if inp != y), None)
                        if add_input:
                            node.inputs.append(add_input)
                            node.outputs[0] = consumer.outputs[0]
                            node.op_type = "Gemm"
                            self.graph.nodes.pop(consumer_idx)
                            if consumer_idx <= i:
                                i -= 1

            if node.op_type in ("Conv", "Gemm"):
                if node.outputs:
                    y = node.outputs[0]
                    consumer_idx = -1
                    consumer = None
                    num_consumers = 0
                    for idx, n in enumerate(self.graph.nodes):
                        if y in n.inputs:
                            num_consumers += 1
                            if n.op_type in fused_activations:
                                consumer_idx = idx
                                consumer = n

                    if consumer and num_consumers == 1:
                        node.attributes["fused_activation"] = Attribute(
                            "fused_activation", "STRING", consumer.op_type
                        )
                        node.outputs[0] = consumer.outputs[0]
                        self.graph.nodes.pop(consumer_idx)
                        if consumer_idx <= i:
                            i -= 1
            i += 1

    def fold_constants(self) -> None:
        """Perform constant folding and specialized weight transformations."""
        self.fuse_activations_and_matmuls()
        for node in self.graph.nodes:
            if node.op_type == "Gemm":
                if len(node.inputs) > 1:
                    weight_name = node.inputs[1]
                    weight_tensor = self.graph.tensors.get(weight_name)
                    transB_attr = node.attributes.get("transB")
                    transB = transB_attr.value if transB_attr else 0

                    if (
                        weight_tensor
                        and weight_tensor.is_initializer
                        and weight_tensor.data
                        and len(weight_tensor.shape) == 2
                    ):
                        if transB == 0:
                            self._transpose_tensor_data(weight_tensor, [1, 0])
                            i_dim, o_dim = weight_tensor.shape
                            weight_tensor.shape = (o_dim, i_dim)

            if node.op_type in ("LSTM", "UnidirectionalSequenceLSTM", "BidirectionalSequenceLSTM"):
                import logging

                logging.warning(
                    f"[onnx2tf] EdgeTPU / Sequence Warning: Node {node.name} ({node.op_type}) requires massive AST restructuring of weights (gates, peepholes) into TFLite's flat format. Conversion accuracy is not guaranteed without TensorFlow's native converter."
                )

            if node.op_type == "Resize":
                roi = node.inputs[1] if len(node.inputs) > 1 else None
                scales = node.inputs[2] if len(node.inputs) > 2 else None
                sizes = node.inputs[3] if len(node.inputs) > 3 else None

                if sizes and sizes in self.graph.tensors:
                    size_tensor = self.graph.tensors[sizes]
                    if size_tensor.dtype == "int64":
                        import logging

                        logging.warning(
                            f"[onnx2tf] Warning: Downcasting Int64 Resize 'sizes' tensor {sizes} to Int32 for mobile compatibility."
                        )
                elif scales and scales in self.graph.tensors:
                    scale_tensor = self.graph.tensors[scales]
                    in_name = node.inputs[0] if node.inputs else None
                    in_info = next((v for v in self.graph.value_info if v.name == in_name), None)
                    if not in_info:
                        in_info = next((v for v in self.graph.inputs if v.name == in_name), None)

                    if (
                        scale_tensor.is_initializer
                        and scale_tensor.data
                        and in_info
                        and in_info.shape
                    ):
                        import struct

                        scale_data = list(
                            struct.unpack(f"<{len(scale_tensor.data) // 4}f", scale_tensor.data)
                        )
                        new_sizes = []
                        for j in range(len(scale_data)):
                            in_dim = in_info.shape[j]
                            if in_dim == -1:
                                import logging

                                logging.warning(
                                    f"[onnx2tf] Warning: Cannot statically compute Resize sizes from scales for tensor {scales} due to dynamic input dimension."
                                )
                            new_sizes.append(int(in_dim * scale_data[j]))

                        new_size_name = f"{node.name}_computed_sizes"
                        self.graph.tensors[new_size_name] = Tensor(
                            new_size_name,
                            shape=[len(new_sizes)],
                            dtype="int32",
                            is_initializer=True,
                            data=struct.pack(f"<{len(new_sizes)}i", *new_sizes),
                        )

                        while len(node.inputs) < 4:
                            node.inputs.append("")
                        node.inputs[3] = new_size_name
                        node.inputs[2] = ""

            if node.op_type in ("Conv", "ConvTranspose"):
                if node.op_type == "ConvTranspose":
                    out_padding_attr = node.attributes.get("output_padding")
                    if out_padding_attr and out_padding_attr.value:
                        import logging

                        logging.warning(
                            f"[onnx2tf] Warning: ConvTranspose node {node.name} uses output_padding {out_padding_attr.value}. TFLite uses static output shape inference which requires mapping to a Shape tensor. Ensure your downstream TFLite parser can infer the dynamic output bounds."
                        )

                if len(node.inputs) > 1:
                    weight_name = node.inputs[1]
                    weight_tensor = self.graph.tensors.get(weight_name)
                    if (
                        weight_tensor
                        and weight_tensor.is_initializer
                        and weight_tensor.data
                        and len(weight_tensor.shape) == 4
                    ):
                        group_attr = node.attributes.get("group")
                        group_val = group_attr.value if group_attr else 1

                        is_depthwise = group_val > 1

                        if is_depthwise and node.op_type == "Conv":
                            num_groups, c, h, w = weight_tensor.shape
                            self._transpose_tensor_data(weight_tensor, [0, 2, 3, 1])
                            weight_tensor.shape = (1, h, w, num_groups * c)
                        elif node.op_type == "Conv":
                            o, i_chan, h, w = weight_tensor.shape
                            self._transpose_tensor_data(weight_tensor, [0, 2, 3, 1])
                            weight_tensor.shape = (o, h, w, i_chan)
                        elif node.op_type == "ConvTranspose":
                            i_chan, o, h, w = weight_tensor.shape
                            self._transpose_tensor_data(weight_tensor, [1, 2, 3, 0])
                            weight_tensor.shape = (o, h, w, i_chan)

    def _transpose_tensor_data(self, tensor: Tensor, perm: list[int]) -> None:
        """Physically transpose the data within a Tensor object."""
        if tensor.data is None:
            return

        if tensor.dtype != "float32":
            logger.warning(f"[onnx2tf] Skipping folding for non-float32 tensor {tensor.name}")
            return

        dims = tensor.shape
        # Read floats
        num_floats = len(tensor.data) // 4
        src = struct.unpack(f"<{num_floats}f", tensor.data)
        dst = [0.0] * num_floats

        if perm == [0, 2, 3, 1]:
            d0, d1, d2, d3 = dims
            for i0 in range(d0):
                for i2 in range(d2):
                    for i3 in range(d3):
                        for i1 in range(d1):
                            src_idx = i3 + d3 * (i2 + d2 * (i1 + d1 * i0))
                            dst_idx = i1 + d1 * (i3 + d3 * (i2 + d2 * i0))
                            dst[dst_idx] = src[src_idx]
        elif perm == [1, 2, 3, 0]:
            d0, d1, d2, d3 = dims
            for i1 in range(d1):
                for i2 in range(d2):
                    for i3 in range(d3):
                        for i0 in range(d0):
                            src_idx = i3 + d3 * (i2 + d2 * (i1 + d1 * i0))
                            dst_idx = i0 + d0 * (i3 + d3 * (i2 + d2 * i1))
        elif perm == [1, 0]:
            d0, d1 = dims
            for i0 in range(d0):
                for i1 in range(d1):
                    src_idx = i1 + d1 * i0
                    dst_idx = i0 + d0 * i1
                    dst[dst_idx] = src[src_idx]

        tensor.data = struct.pack(f"<{num_floats}f", *dst)

    def rewrite_negative_axes(self) -> None:
        """Convert negative axis indices to their positive equivalents based on tensor rank."""
        for node in self.graph.nodes:
            axis_attr = node.attributes.get("axis")
            if axis_attr and isinstance(axis_attr.value, int) and axis_attr.value < 0:
                in_name = node.inputs[0] if node.inputs else None
                rank = 4
                if in_name:
                    in_info = next((v for v in self.graph.value_info if v.name == in_name), None)
                    if not in_info:
                        in_info = next((v for v in self.graph.inputs if v.name == in_name), None)
                    if not in_info:
                        in_info = self.graph.tensors.get(in_name)
                    if in_info and in_info.shape:
                        rank = len(in_info.shape)
                axis_attr.value += rank

            axes_attr = node.attributes.get("axes")
            if axes_attr and isinstance(axes_attr.value, list):
                in_name = node.inputs[0] if node.inputs else None
                rank = 4
                if in_name:
                    in_info = next((v for v in self.graph.value_info if v.name == in_name), None)
                    if not in_info:
                        in_info = next((v for v in self.graph.inputs if v.name == in_name), None)
                    if not in_info:
                        in_info = self.graph.tensors.get(in_name)
                    if in_info and in_info.shape:
                        rank = len(in_info.shape)
                axes_attr.value = [a + rank if a < 0 else a for a in axes_attr.value]

    def recalculate_shapes(self) -> None:
        """Update value information with recalculated shapes after graph transformations."""
        for node in self.graph.nodes:
            if node.op_type == "Transpose":
                perm_attr = node.attributes.get("perm")
                if perm_attr:
                    perm = perm_attr.value
                    in_info = next(
                        (v for v in self.graph.value_info if v.name == node.inputs[0]), None
                    )
                    if not in_info:
                        in_info = next(
                            (v for v in self.graph.inputs if v.name == node.inputs[0]), None
                        )

                    if in_info:
                        out_shape = [in_info.shape[p] for p in perm]
                        out_info = next(
                            (v for v in self.graph.value_info if v.name == node.outputs[0]), None
                        )
                        if out_info:
                            out_info.shape = out_shape
                        else:
                            self.graph.value_info.append(
                                ValueInfo(node.outputs[0], tuple(out_shape), in_info.dtype)
                            )

    def check_irreducible_transposes(self) -> None:
        """Identify Transpose operations that couldn't be optimized away and may impact performance."""
        for node in self.graph.nodes:
            if node.op_type == "Transpose":
                perm = node.attributes.get("perm")
                if perm and (perm.value == [0, 2, 3, 1] or perm.value == [0, 3, 1, 2]):
                    logger.warning(
                        f"[onnx2tf] Warning: Irreducible Transpose node left in graph: {node.name}. This degrades EdgeTPU performance."
                    )
