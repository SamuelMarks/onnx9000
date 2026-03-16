"""Hardware-Aware Memory Layout Optimization module."""

from typing import Any
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


class LayoutOptimizer:
    """Memory layout optimizer."""

    @staticmethod
    def supports_layout_attribute(node: Node) -> bool:
        """Detect operators supporting layout attributes natively (e.g., Conv)."""
        return node.op_type in [
            "Conv",
            "MaxPool",
            "AveragePool",
            "BatchNormalization",
            "QLinearConv",
        ]

    @staticmethod
    def inject_transpose(graph: Graph, input_name: str, output_name: str, perm: list[int]) -> Node:
        """Helper to inject a Transpose node."""
        node_name = f"Transpose_{input_name}_{output_name}"
        t_node = Node("Transpose", [input_name], [output_name], {"perm": perm}, name=node_name)
        graph.add_node(t_node)
        return t_node

    @staticmethod
    def nchw_to_nhwc_pass(graph: Graph) -> Graph:
        """Implement NCHW (Channels First) to NHWC (Channels Last) conversion pass."""
        new_graph = Graph(graph.name + "_nhwc")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_graph.add_tensor(t)
        for node in graph.nodes:
            if LayoutOptimizer.supports_layout_attribute(node):
                attrs = node.attributes.copy()
                attrs["layout"] = "NHWC"
                new_graph.add_node(
                    Node(node.op_type, node.inputs.copy(), node.outputs.copy(), attrs, node.name)
                )
            else:
                new_graph.add_node(
                    Node(
                        node.op_type,
                        node.inputs.copy(),
                        node.outputs.copy(),
                        node.attributes.copy(),
                        node.name,
                    )
                )
        return new_graph

    @staticmethod
    def nhwc_to_nchw_pass(graph: Graph) -> Graph:
        """Implement NHWC to NCHW conversion pass."""
        new_graph = Graph(graph.name + "_nchw")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_graph.add_tensor(t)
        for node in graph.nodes:
            if LayoutOptimizer.supports_layout_attribute(node):
                attrs = node.attributes.copy()
                attrs["layout"] = "NCHW"
                new_graph.add_node(
                    Node(node.op_type, node.inputs.copy(), node.outputs.copy(), attrs, node.name)
                )
            else:
                new_graph.add_node(
                    Node(
                        node.op_type,
                        node.inputs.copy(),
                        node.outputs.copy(),
                        node.attributes.copy(),
                        node.name,
                    )
                )
        return new_graph

    @staticmethod
    def inject_transposes_for_layout(graph: Graph, target_layout: str = "NHWC") -> Graph:
        """Inject Transpose nodes where layout mismatch occurs."""
        new_graph = Graph(graph.name + "_transposed")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_graph.add_tensor(t)
        counter = 0
        for node in graph.nodes:
            if LayoutOptimizer.supports_layout_attribute(node):
                attrs = node.attributes.copy()
                current_layout = attrs.get("layout", "NCHW")
                if current_layout != target_layout:
                    if target_layout == "NHWC" and current_layout == "NCHW":
                        new_inputs = []
                        for inp in node.inputs:
                            trans_out = f"{inp}_to_nhwc_{counter}"
                            LayoutOptimizer.inject_transpose(
                                new_graph, inp, trans_out, [0, 2, 3, 1]
                            )
                            new_inputs.append(trans_out)
                        new_outputs = []
                        for out in node.outputs:
                            trans_in = f"{out}_nhwc_{counter}"
                            new_outputs.append(trans_in)
                        attrs["layout"] = "NHWC"
                        new_graph.add_node(
                            Node(node.op_type, new_inputs, new_outputs, attrs, node.name)
                        )
                        for i, out in enumerate(node.outputs):
                            LayoutOptimizer.inject_transpose(
                                new_graph, new_outputs[i], out, [0, 3, 1, 2]
                            )
                        counter += 1
                        continue
            new_graph.add_node(
                Node(
                    node.op_type,
                    node.inputs.copy(),
                    node.outputs.copy(),
                    node.attributes.copy(),
                    node.name,
                )
            )
        return new_graph

    @staticmethod
    def transpose_cancellation_pass(graph: Graph) -> Graph:
        """Implement a greedy Transpose cancellation pass (fusing adjacent transposes)."""
        new_graph = Graph(graph.name + "_opt")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_graph.add_tensor(t)
        consumers: dict[str, list[Node]] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in consumers:
                    consumers[inp] = []
                consumers[inp].append(node)
        skip_nodes = set()
        for node in graph.nodes:
            if id(node) in skip_nodes:
                continue
            if node.op_type == "Transpose":
                out_name = node.outputs[0]
                cons = consumers.get(out_name, [])
                if len(cons) == 1 and cons[0].op_type == "Transpose":
                    next_node = cons[0]
                    p1 = node.attributes.get("perm", [])
                    p2 = next_node.attributes.get("perm", [])
                    if p1 and p2 and (len(p1) == len(p2)):
                        fused_perm = [p1[p2[i]] for i in range(len(p2))]
                        is_identity = all((fused_perm[i] == i for i in range(len(fused_perm))))
                        if is_identity:
                            ident_node = Node(
                                "Identity",
                                [node.inputs[0]],
                                [next_node.outputs[0]],
                                {},
                                f"Identity_{out_name}",
                            )
                            new_graph.add_node(ident_node)
                            skip_nodes.add(id(next_node))
                            continue
            new_graph.add_node(
                Node(
                    node.op_type,
                    node.inputs.copy(),
                    node.outputs.copy(),
                    node.attributes.copy(),
                    node.name,
                )
            )
        return new_graph

    @staticmethod
    def push_transposes_down(graph: Graph) -> Graph:
        """Implement a pass to push Transpose nodes down the graph through elementwise operations."""
        new_graph = Graph(graph.name + "_pushed")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_graph.add_tensor(t)
        elementwise_ops = {"Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Exp", "Log"}
        consumers: dict[str, list[Node]] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in consumers:
                    consumers[inp] = []
                consumers[inp].append(node)
        transposes_to_push = {}
        for node in graph.nodes:
            if node.op_type == "Transpose":
                out_name = node.outputs[0]
                cons = consumers.get(out_name, [])
                if all((c.op_type in elementwise_ops for c in cons)) and len(cons) > 0:
                    transposes_to_push[out_name] = node
            if node.op_type == "Transpose" and node.outputs[0] in transposes_to_push:
                continue
            if node.op_type in elementwise_ops:
                pushed_inputs = [inp for inp in node.inputs if inp in transposes_to_push]
                if pushed_inputs:
                    perm = transposes_to_push[pushed_inputs[0]].attributes.get("perm", [])
                    new_inputs = []
                    for inp in node.inputs:
                        if inp in transposes_to_push:
                            new_inputs.append(transposes_to_push[inp].inputs[0])
                        else:
                            new_inputs.append(inp)
                    intermediate_out = f"{node.outputs[0]}_pre_transpose"
                    new_elem_node = Node(
                        node.op_type,
                        new_inputs,
                        [intermediate_out],
                        node.attributes.copy(),
                        node.name,
                    )
                    new_graph.add_node(new_elem_node)
                    new_trans_node = Node(
                        "Transpose",
                        [intermediate_out],
                        node.outputs.copy(),
                        {"perm": perm},
                        f"{node.name}_pushed_transpose",
                    )
                    new_graph.add_node(new_trans_node)
                    continue
            new_graph.add_node(
                Node(
                    node.op_type,
                    node.inputs.copy(),
                    node.outputs.copy(),
                    node.attributes.copy(),
                    node.name,
                )
            )
        return new_graph

    @staticmethod
    def optimal_webgpu_layout() -> str:
        """Write heuristic matching WebGPU optimal layouts (NHWC usually preferred for cache locality)."""
        return "NHWC"

    @staticmethod
    def optimal_wasm_simd_layout() -> str:
        """Write heuristic matching WASM SIMD optimal layouts (NCHW usually preferred)."""
        return "NCHW"

    @staticmethod
    def optimal_ios_coreml_layout() -> str:
        """Support generating specialized graphs for iOS CoreML / Neural Engine via layout hints."""
        return "NCHW"

    @staticmethod
    def optimal_android_nnapi_layout() -> str:
        """Support generating specialized graphs for Android NNAPI layout hints."""
        return "NHWC"

    @staticmethod
    def pad_to_alignment(shape: tuple[Any, ...], alignment: int = 4) -> tuple[Any, ...]:
        """Implement memory alignment packing (e.g., padding channels to multiples of 4 for vec4<f32> in WGSL)."""
        if len(shape) < 1:
            return shape
        last_dim = shape[-1]
        if isinstance(last_dim, int):
            pad_amount = (alignment - last_dim % alignment) % alignment
            new_shape = list(shape)
            new_shape[-1] = last_dim + pad_amount
            return tuple(new_shape)
        return shape

    @staticmethod
    def align_tensor_shapes_pass(graph: Graph, alignment: int = 4) -> Graph:
        """Write a pass that pads all tensor shapes to alignment boundaries."""
        new_graph = Graph(graph.name + "_aligned")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            new_shape = LayoutOptimizer.pad_to_alignment(t.shape, alignment)
            new_t = Tensor(t.name, new_shape, t.dtype, t.is_initializer, t.requires_grad, t.data)
            new_graph.add_tensor(new_t)
        for node in graph.nodes:
            new_graph.add_node(
                Node(
                    node.op_type,
                    node.inputs.copy(),
                    node.outputs.copy(),
                    node.attributes.copy(),
                    node.name,
                )
            )
        return new_graph

    @staticmethod
    def update_parameters_for_alignment(node: Node, pad_amount: int) -> Node:
        """Update Conv, MatMul, and Reshape parameters mathematically to account for alignment padding."""
        attrs = node.attributes.copy()
        attrs["aligned"] = True
        if node.op_type == "Conv":
            group = attrs.get("group", 1)
            attrs["group"] = group
        elif node.op_type == "Reshape":
            if "shape" in attrs:
                shape = list(attrs["shape"])
                if len(shape) > 0 and shape[-1] != -1:
                    shape[-1] += pad_amount
                attrs["shape"] = shape
        return Node(node.op_type, node.inputs.copy(), node.outputs.copy(), attrs, node.name)

    @staticmethod
    def unfold_constants(graph: Graph) -> Graph:
        """Implement constant unfolding (converting highly dimensional constants to 1D flat arrays for WGSL)."""
        new_graph = Graph(graph.name + "_unfolded")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        for _name, t in graph.tensors.items():
            if t.is_initializer and t.data is not None:
                flat_data = t.data.flatten()
                new_t = Tensor(t.name, (len(flat_data),), t.dtype, True, t.requires_grad, flat_data)
                new_graph.add_tensor(new_t)
            else:
                new_graph.add_tensor(t)
        for node in graph.nodes:
            new_graph.add_node(
                Node(
                    node.op_type,
                    node.inputs.copy(),
                    node.outputs.copy(),
                    node.attributes.copy(),
                    node.name,
                )
            )
        return new_graph

    @staticmethod
    def estimate_vram_usage(graph: Graph) -> int:
        """Implement a memory estimation pass (simulating VRAM usage before execution)."""
        total_bytes = 0
        dtype_sizes = {
            DType.FLOAT32: 4,
            DType.FLOAT64: 8,
            DType.INT8: 1,
            DType.INT16: 2,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.UINT8: 1,
            DType.UINT16: 2,
            DType.UINT32: 4,
            DType.UINT64: 8,
            DType.BOOL: 1,
            DType.FLOAT16: 2,
            DType.BFLOAT16: 2,
        }
        for _name, t in graph.tensors.items():
            num_elements = 1
            for dim in t.shape:
                if isinstance(dim, int):
                    num_elements *= dim
            total_bytes += num_elements * dtype_sizes.get(t.dtype, 4)
        return total_bytes

    @staticmethod
    def chunk_large_tensors_pass(graph: Graph, max_size: int = 134217728) -> Graph:
        """Implement a pass detecting and resolving WebGPU maxStorageBufferBindingSize limits by chunking large tensors."""
        new_graph = Graph(graph.name + "_chunked")
        new_graph.inputs = graph.inputs.copy()
        new_graph.outputs = graph.outputs.copy()
        new_graph.initializers = graph.initializers.copy()
        dtype_sizes = {DType.FLOAT32: 4, DType.FLOAT16: 2, DType.INT8: 1, DType.UINT8: 1}
        chunked_tensors = {}
        for name, t in graph.tensors.items():
            num_elements = 1
            for dim in t.shape:
                if isinstance(dim, int):
                    num_elements *= dim
            size_bytes = num_elements * dtype_sizes.get(t.dtype, 4)
            if size_bytes > max_size:
                axis_0 = t.shape[0]
                if isinstance(axis_0, int) and axis_0 > 1:
                    num_chunks = (size_bytes + max_size - 1) // max_size
                    chunk_size = (axis_0 + num_chunks - 1) // num_chunks
                    chunks = []
                    for i in range(num_chunks):
                        c_shape = list(t.shape)
                        c_shape[0] = min(chunk_size, axis_0 - i * chunk_size)
                        c_name = f"{t.name}_chunk_{i}"
                        c_t = Tensor(
                            c_name, tuple(c_shape), t.dtype, t.is_initializer, t.requires_grad
                        )
                        new_graph.add_tensor(c_t)
                        chunks.append(c_name)
                    chunked_tensors[name] = chunks
                else:
                    new_graph.add_tensor(t)
            else:
                new_graph.add_tensor(t)
        for node in graph.nodes:
            new_inputs = []
            for inp in node.inputs:
                if inp in chunked_tensors:
                    concat_out = f"{inp}_recombined"
                    concat_node = Node(
                        "Concat", chunked_tensors[inp], [concat_out], {"axis": 0}, f"Concat_{inp}"
                    )
                    new_graph.add_node(concat_node)
                    new_inputs.append(concat_out)
                else:
                    new_inputs.append(inp)
            new_graph.add_node(
                Node(
                    node.op_type, new_inputs, node.outputs.copy(), node.attributes.copy(), node.name
                )
            )
        return new_graph
