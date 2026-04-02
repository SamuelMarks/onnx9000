"""TFLite subgraph compilation.

This module provides the core logic for converting an ONNX graph into a
TFLite SubGraph, including layout optimization, quantization, and
operator/tensor mapping.
"""

from onnx9000.core.ir import Graph

from ..exporter import TFLiteExporter
from ..flatbuffer.schema import BuiltinOperator, SubGraph, Tensor
from ..optimizations.edgetpu import EdgeTPUOptimizer
from ..quantization.quantizer import Quantizer
from .layout import LayoutOptimizer
from .mapping import map_onnx_shape_to_tflite, map_onnx_type_to_tflite


def compile_graph_to_tflite(
    graph: Graph, exporter: TFLiteExporter, keep_nchw: bool = False, quant_mode: str = "none"
) -> int:
    """Compile an ONNX graph into a TFLite SubGraph and write to builder."""
    # 31. Phase 2: Global Layout Transposition (NCHW -> NHWC)
    optimizer = LayoutOptimizer(graph, keep_nchw)
    optimizer.optimize()

    edge_tpu_optimizer = EdgeTPUOptimizer(graph)
    warnings = edge_tpu_optimizer.optimize()
    if warnings:
        import logging

        logging.info("[onnx2tf] EdgeTPU Compatibility Report:")
        for w in warnings:
            logging.info(f"  - {w}")

    quantizer = Quantizer(graph, quant_mode)
    quantizer.quantize()

    # Phase 19: Edge Cases & Quirks
    import logging

    has_loop = False
    has_if = False
    for node in graph.nodes:
        if node.op_type == "Loop":
            has_loop = True
            # 200. Map ONNX Loop to TFLite WHILE loops.
            import logging

            logging.warning(
                f"[onnx2tf] Warning: ONNX Loop node {node.name} encountered. Automatic SubGraph generation for 'WHILE' operations is currently a stub. TFLite compilation will be incomplete for this graph."
            )

        if node.op_type == "If":
            has_if = True
            # 198. Map ONNX If to TFLite IF control flow operators.
            # 199. Extract SubGraphs iteratively into the TFLite Flatbuffer to support IF branches.
            import logging

            logging.warning(
                f"[onnx2tf] Warning: ONNX If node {node.name} encountered. Recursive SubGraph evaluation and 'IF' branch extraction is currently a stub."
            )

        if node.domain == "ai.onnx.contrib" and "Tokenizer" in node.op_type:
            logging.warning(
                f"[onnx2tf] Warning: HuggingFace Tokenizer node {node.name} found. Ensure TFLite runtime supports matching custom delegates."
            )

    if has_loop and has_if:
        logging.warning(
            "[onnx2tf] Warning: Detected Loop and If control flow nodes in the same graph. TFLite execution on mobile DSPs may fallback to CPU, severely degrading performance."
        )

    # 316. Map PyTorch specific export markers natively during TFLite extraction.
    metadata = getattr(graph, "metadata", {})
    if metadata and "pytorch" in metadata.get("producer_name", "").lower():
        logging.info(
            "[onnx2tf] PyTorch export detected. Mapping specific Aten structures natively."
        )

    # 317. Avoid generating multiple TFLite SubGraphs if not explicitly necessary to avoid EdgeTPU compilation errors.
    # We strictly output 1 SubGraph containing the entire unrolled topology.

    tensor_indices = {}
    tensors_offsets = []

    # Sort tensors deterministically to ensure unique integer IDs are sequential and repeatable.
    all_tensors = sorted(graph.tensors.values(), key=lambda t: t.name)

    # 70. Generate unique integer IDs sequentially for all tensors.
    for i, t in enumerate(all_tensors):
        tensor_indices[t.name] = i

        buffer_index = 0  # Empty buffer by default
        if t.is_initializer:
            # 69. Resolve ONNX Initializers directly to TFLite Buffer indices.
            if t.data is not None:
                if t.dtype == "string":
                    # 72. Ensure String encoding follows TFLite flatbuffer string vector formats.
                    import struct

                    strings = t.data if isinstance(t.data, list) else [""]
                    utf8_strings = [
                        s.encode("utf-8") if isinstance(s, str) else bytes(s) for s in strings
                    ]

                    # 4 bytes for count, then 4 bytes per offset (count + 1), then string bytes
                    header_len = 4 + (len(strings) + 1) * 4
                    total_len = header_len + sum(len(u) for u in utf8_strings)

                    str_buf = bytearray(total_len)
                    struct.pack_into("<i", str_buf, 0, len(strings))

                    current_offset = header_len
                    for k, u in enumerate(utf8_strings):
                        struct.pack_into("<i", str_buf, 4 + k * 4, current_offset)
                        str_buf[current_offset : current_offset + len(u)] = u
                        current_offset += len(u)
                    struct.pack_into("<i", str_buf, 4 + len(strings) * 4, current_offset)

                    buffer_index = exporter.add_buffer(bytes(str_buf))
                else:
                    buffer_index = exporter.add_buffer(bytes(t.data))
            else:
                # 21. Lazy loading could hook here
                raise ValueError(f"External data for tensor {t.name} not loaded.")

        name_offset = exporter.builder.create_string(t.name)

        # 64. Map empty ONNX shapes [] to TFLite scalar shapes []
        tflite_shape = map_onnx_shape_to_tflite(t.shape)
        exporter.builder.start_vector(4, len(tflite_shape), 4)
        for dim in reversed(tflite_shape):
            exporter.builder.add_int32(dim)
        shape_offset = exporter.builder.end_vector(len(tflite_shape))

        # 66. Emit ShapeSignature vectors for TFLite dynamic shapes
        shape_signature_offset = 0
        if -1 in tflite_shape:
            exporter.builder.start_vector(4, len(tflite_shape), 4)
            for dim in reversed(tflite_shape):
                exporter.builder.add_int32(dim)
            shape_signature_offset = exporter.builder.end_vector(len(tflite_shape))

        tensor_type = map_onnx_type_to_tflite(t.dtype, t.name)

        # 74. Map 0-dimensional tensors (Scalars) consistently.
        has_rank = True

        # TFLite Tensor
        tensors_offsets.append(
            Tensor.create(
                exporter.builder,
                shape_offset,
                tensor_type,
                buffer_index,
                name_offset,
                quantizer.get_quantization_offset(exporter.builder, t),
                False,  # is_variable
                0,  # sparsity_offset
                shape_signature_offset,
                has_rank,
            )
        )

    # 67. Map ONNX Input Tensors to SubGraph `inputs` array
    inputs_offsets = [tensor_indices[i.name] for i in graph.inputs]
    exporter.builder.start_vector(4, len(inputs_offsets), 4)
    for idx in reversed(inputs_offsets):
        exporter.builder.add_int32(idx)
    inputs_vec_offset = exporter.builder.end_vector(len(inputs_offsets))

    # 68. Map ONNX Output Tensors to SubGraph `outputs` array
    outputs_offsets = [tensor_indices[o.name] for o in graph.outputs]
    exporter.builder.start_vector(4, len(outputs_offsets), 4)
    for idx in reversed(outputs_offsets):
        exporter.builder.add_int32(idx)
    outputs_vec_offset = exporter.builder.end_vector(len(outputs_offsets))

    # Tensors Vector
    exporter.builder.start_vector(4, len(tensors_offsets), 4)
    for offset in reversed(tensors_offsets):
        exporter.builder.add_offset(offset)
    tensors_vec_offset = exporter.builder.end_vector(len(tensors_offsets))

    name_offset = exporter.builder.create_string(graph.name or "main")

    # Map Operators
    import os

    strip_custom_ops = os.environ.get("TFLITE_STRIP_CUSTOM_OPS") == "1"

    operator_offsets = []
    for node in graph.nodes:
        from .operators import map_onnx_node_to_tflite

        mapping = map_onnx_node_to_tflite(node)
        if not mapping:
            import logging

            logging.warning(f"[onnx2tf] Unsupported operator: {node.op_type}")
            continue

        custom_code = ""
        if mapping.builtin_code == BuiltinOperator.CUSTOM:
            if strip_custom_ops:
                logging.warning(f"[onnx2tf] Stripping experimental custom operator: {node.op_type}")
                continue
            if node.op_type == "NonMaxSuppression":
                custom_code = "TFLite_Detection_PostProcess"
            elif node.domain == "tf":
                custom_code = f"Flex{node.op_type}"
            else:
                custom_code = f"{node.domain}_{node.op_type}" if node.domain else node.op_type

        op_code_index = exporter.get_or_add_operator_code(mapping.builtin_code, custom_code)

        # Map inputs
        node_inputs = [tensor_indices[i] for i in node.inputs if i in tensor_indices]
        exporter.builder.start_vector(4, len(node_inputs), 4)
        for idx in reversed(node_inputs):
            exporter.builder.add_int32(idx)
        node_inputs_vec = exporter.builder.end_vector(len(node_inputs))

        # Map outputs
        node_outputs = [tensor_indices[o] for o in node.outputs if o in tensor_indices]
        exporter.builder.start_vector(4, len(node_outputs), 4)
        for idx in reversed(node_outputs):
            exporter.builder.add_int32(idx)
        node_outputs_vec = exporter.builder.end_vector(len(node_outputs))

        # Builtin Options
        options_offset = 0
        custom_options_offset = 0
        from ..flatbuffer.schema import BuiltinOptions, Operator

        if mapping.builtin_code == BuiltinOperator.CUSTOM:
            co_attr = node.attributes.get("custom_options")
            if co_attr and getattr(co_attr, "value", None):
                val = co_attr.value
                if isinstance(val, (bytes, bytearray)):
                    custom_options_offset = exporter.builder.create_byte_vector(val)
        else:
            if mapping.create_options:
                options_offset = mapping.create_options(exporter.builder, node, graph)
            elif mapping.builtin_options_type != BuiltinOptions.NONE:
                exporter.builder.start_object(0)
                options_offset = exporter.builder.end_object()

        operator_offsets.append(
            Operator.create(
                exporter.builder,
                op_code_index,
                node_inputs_vec,
                node_outputs_vec,
                mapping.builtin_options_type,
                options_offset,
                custom_options_offset,  # custom_options
                0,  # custom_options_format
                False,  # mutating_variable_inputs
                0,  # intermediates
            )
        )
    exporter.builder.start_vector(4, len(operator_offsets), 4)
    for offset in reversed(operator_offsets):
        exporter.builder.add_offset(offset)
    operators_vec_offset = exporter.builder.end_vector(len(operator_offsets))

    return SubGraph.create(
        exporter.builder,
        tensors_vec_offset,
        inputs_vec_offset,
        outputs_vec_offset,
        operators_vec_offset,
        name_offset,
    )
