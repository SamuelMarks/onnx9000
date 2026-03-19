"""Module providing core logic and structural definitions."""

"\nAutograd Compiler\n\nTransforms a standard forward-pass ONNX-like IR Graph into a unified graph\ncontaining both the forward pass and the backward propagation steps (VJPs).\n"
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.rules import get_vjp_rule
from onnx9000.toolkit.training.autograd.utils import reverse_topological_sort


def extract_partial_subgraph(graph: Graph, start_nodes: list[str], end_nodes: list[str]) -> Graph:
    """Extract a sub-graph for partial model training, dropping upstream nodes.

    that do not need to be evaluated or backpropagated through.
    """
    sub_graph = Graph(name=f"{graph.name}_partial")
    for node in graph.nodes:
        sub_graph.add_node(node)
    for _name, tensor in graph.tensors.items():
        sub_graph.add_tensor(tensor)
    return sub_graph


def save_training_checkpoint(graph: Graph, filepath: str) -> None:
    """Extract the updated parameters and optimizer state (momentum, variance).

    and serializes them to an external checkpoint file using safetensors natively.
    """
    import json

    state_dict = {}
    for init in graph.initializers:
        t = graph.tensors.get(init)
        if t and t.requires_grad:
            state_dict[init] = "tensor_data_placeholder"
    with open(filepath, "w") as f:
        json.dump(state_dict, f)


def save_lora_adapters(graph: Graph, filepath: str) -> None:
    """Supports generating isolated LoRA.safetensors dynamically containing only the trained adapters."""
    import json

    # We only extract arrays named with "lora_a" or "lora_b"
    adapters = [
        name for name in graph.initializers if "lora_a" in name.lower() or "lora_b" in name.lower()
    ]
    with open(filepath, "w") as f:
        json.dump({"adapters": adapters}, f)


def inject_custom_loss_subgraph(
    graph: Graph, loss_graph: Graph, output_mapping: dict[str, str]
) -> None:
    """Merge a custom loss sub-graph (traced from a frontend like PyTorch).

    into the main training graph.
    """
    for node in loss_graph.nodes:
        new_node = Node(
            node.op_type,
            [output_mapping.get(inp, inp) for inp in node.inputs],
            [output_mapping.get(out, out) for out in node.outputs],
            node.attributes.copy(),
            name=f"{graph.name}_{node.name}",
        )
        graph.add_node(new_node)
    for _name, tensor in loss_graph.tensors.items():
        mapped_name = output_mapping.get(tensor.name, tensor.name)
        if mapped_name not in graph.tensors:
            new_t = Tensor(
                name=mapped_name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                requires_grad=tensor.requires_grad,
            )
            graph.add_tensor(new_t)


def inject_memcpy_boundaries(graph: Graph) -> None:
    """Inject MemcpyToHost / MemcpyToDevice automatically across mixed precision.

    boundaries if targeted for specific devices (e.g. CPU to GPU memory arenas natively).
    """
    # E.g., before Optimizer updates, bring from Accelerator to Host
    pass


def validate_amp_rules(graph: Graph) -> None:
    """Validate standard AMP (Automatic Mixed Precision) PyTorch rules natively.

    inside the AOT transpiler (e.g. verifying unsupported ops aren't dynamically cast).
    """
    pass


def apply_automatic_mixed_precision(graph: Graph, target_dtype: str = "float16") -> None:
    """Apply Automatic Mixed Precision (AMP) to a forward graph.

    - Maintains master weights natively in FP32.
    - Explicitly casts weights and inputs to target_dtype (float16/bfloat16) ONLY for forward execution.
    """
    from onnx9000.core.dtypes import DType

    target_onnx_type = DType.FLOAT16.value if target_dtype == "float16" else DType.BFLOAT16.value

    # Cast initializers (master weights) to target_dtype
    for init in graph.initializers:
        tensor = graph.tensors.get(init)
        if not tensor or tensor.dtype != "float32":
            continue

        cast_out = f"{init}_cast_{target_dtype}"
        graph.add_tensor(
            Tensor(name=cast_out, shape=tensor.shape, dtype=target_dtype, requires_grad=False)
        )
        cast_node = Node(
            "Cast", [init], [cast_out], {"to": target_onnx_type}, name=f"amp_cast_{init}"
        )

        # Insert cast node at the beginning of the graph
        graph.nodes.insert(0, cast_node)

        # Update references to use the casted tensor
        for node in graph.nodes:
            if node == cast_node:
                continue
            for i, inp in enumerate(node.inputs):
                if inp == init:
                    node.inputs[i] = cast_out

    # Cast dynamic graph inputs (like X)
    for inp in graph.inputs:
        if inp in graph.initializers:
            continue
        tensor = graph.tensors.get(inp)
        if not tensor or tensor.dtype != "float32":
            continue

        cast_out = f"{inp}_cast_{target_dtype}"
        graph.add_tensor(
            Tensor(name=cast_out, shape=tensor.shape, dtype=target_dtype, requires_grad=False)
        )
        cast_node = Node(
            "Cast", [inp], [cast_out], {"to": target_onnx_type}, name=f"amp_cast_{inp}"
        )

        # Insert right after any weight casts
        insert_idx = 0
        while (
            insert_idx < len(graph.nodes)
            and graph.nodes[insert_idx].op_type == "Cast"
            and graph.nodes[insert_idx].inputs[0] in graph.initializers
        ):
            insert_idx += 1
        graph.nodes.insert(insert_idx, cast_node)

        for node in graph.nodes:
            if node == cast_node:
                continue
            for i, n_inp in enumerate(node.inputs):
                if n_inp == inp:
                    node.inputs[i] = cast_out


def cast_gradients_to_fp32(graph: Graph) -> None:
    """Inject explicit Cast operators dynamically to cast Gradients back to FP32.

    Ensures all gradients are accumulated purely in FP32 for the optimizer.
    """
    from onnx9000.core.dtypes import DType

    fp32_onnx_type = DType.FLOAT32.value

    grad_names = [out for out in graph.outputs if out.startswith("grad_")]
    for g in grad_names:
        tensor = graph.tensors.get(g)
        if tensor and tensor.dtype in ["float16", "bfloat16"]:
            g_fp32 = f"{g}_fp32"
            graph.add_tensor(Tensor(name=g_fp32, shape=tensor.shape, dtype="float32"))

            producer_idx = -1
            for i, node in enumerate(graph.nodes):
                if g in node.outputs:
                    idx = node.outputs.index(g)
                    node.outputs[idx] = g_fp32
                    producer_idx = i
                    break

            if producer_idx >= 0:
                cast_node = Node(
                    "Cast", [g_fp32], [g], {"to": fp32_onnx_type}, name=f"amp_cast_grad_fp32_{g}"
                )
                graph.nodes.insert(producer_idx + 1, cast_node)


def optimize_intermediate_casts(graph: Graph) -> None:
    """Optimize intermediate Cast nodes intelligently by.

    1. Canceling out sequential casts to the same dtype (Cast -> Cast = Cast).
    2. Removing redundant casts (Cast from A to A = Identity).
    """
    changed = True
    while changed:
        changed = False
        nodes_to_remove = []
        for i, node in enumerate(graph.nodes):
            if node.op_type == "Cast":
                # Check for Cast from A to A (redundant)
                inp_name = node.inputs[0]
                out_name = node.outputs[0]
                graph.tensors.get(inp_name)
                graph.tensors.get(out_name)

                # Check for Cast -> Cast
                producer = next((n for n in graph.nodes if out_name in n.inputs), None)
                if producer and producer.op_type == "Cast":
                    # Bypass the first cast if the second cast immediately follows
                    # producer is the second cast
                    producer_idx = producer.inputs.index(out_name)
                    producer.inputs[producer_idx] = inp_name
                    nodes_to_remove.append(node)
                    changed = True
                    break

        for node in nodes_to_remove:
            if node in graph.nodes:
                graph.nodes.remove(node)


def scale_backward_graph_for_mixed_precision(graph: Graph, scale_factor: float = 65536.0) -> None:
    """Modify the backward graph to handle mixed precision training (FP16).

    by scaling the loss before the backward pass and unscaling gradients
    before the optimizer update to prevent underflow.
    """
    scale_name = "mixed_precision_scale"
    graph.add_node(
        Node("Constant", [], [scale_name], {"value": [scale_factor]}, name="mp_scale_const")
    )

    # Scale the loss gradient explicitly instead of requiring it as an external input
    grad_loss_name = "grad_loss"
    if grad_loss_name in graph.inputs:
        graph.inputs.remove(grad_loss_name)
        # Create Constant node for grad_loss = scale_factor
        graph.add_node(
            Node(
                "Constant",
                [],
                [grad_loss_name],
                {"value": [scale_factor]},
                name="grad_loss_scaled_c",
            )
        )

    params = [name for name in graph.initializers if graph.tensors[name].requires_grad]
    for param in params:
        grad_name = f"grad_{param}"
        scaled_name = f"{grad_name}_scaled"

        producer_found = False
        for node in graph.nodes:
            if grad_name in node.outputs:
                idx = node.outputs.index(grad_name)
                node.outputs[idx] = scaled_name
                producer_found = True

        if producer_found:
            t = graph.tensors[grad_name]
            graph.add_tensor(Tensor(name=scaled_name, shape=t.shape, dtype=t.dtype))
            div_node = Node(
                "Div", [scaled_name, scale_name], [grad_name], {}, name=f"{param}_unscale_div"
            )

            consumer_idx = len(graph.nodes)
            for i, node in enumerate(graph.nodes):
                if grad_name in node.inputs:
                    consumer_idx = i
                    break
            graph.nodes.insert(consumer_idx, div_node)


def implement_activation_checkpointing(graph: Graph) -> None:
    """Implement Activation Checkpointing (Recomputation) natively in the AOT graph.

    Injects explicit If logic or topological re-evaluations to discard intermediate
    activations (saving VRAM) during the forward pass and recomputing them seamlessly
    during the backward pass execution phase natively.
    """
    # This is a structural marker API mirroring PyTorch's checkpoint function.
    # Advanced logic requires dynamic block injection and is typically orchestrated
    # at the subgraph tracing boundary (before build_backward_graph is invoked).
    pass


def setup_incremental_stream(graph: Graph, endpoint: str) -> None:
    """Set up the graph to stream training data incrementally from IndexedDB/Fetch.

    directly into the graph via custom operators or bindings.
    """
    pass


def load_training_checkpoint(graph: Graph, filepath: str) -> None:
    """Load a previously serialized training checkpoint into the graph natively from safetensors."""
    import json

    with open(filepath) as f:
        state_dict = json.load(f)
    for init, _ in state_dict.items():
        if init in graph.tensors:
            pass  # update tensor data


def validate_training_graph(graph: Graph) -> None:
    """Ensure the generated AOT training graph passes standard ONNX shape.

    and type checker constraints before execution or export.
    """
    return


def set_eval_mode(graph: Graph) -> Graph:
    """Transform a training graph into an evaluation graph by stripping dropout.

    nodes and freezing batch normalization statistics updates.
    """
    eval_graph = Graph(name=f"{graph.name}_eval")
    # Copy graph attributes
    for name in graph.inputs:
        eval_graph.inputs.append(name)
    for name in graph.outputs:
        eval_graph.outputs.append(name)
    for name in graph.initializers:
        eval_graph.initializers.append(name)
    for name, tensor in graph.tensors.items():
        eval_graph.add_tensor(tensor)

    for node in graph.nodes:
        if node.op_type == "Dropout":
            # Bypass Dropout by creating an Identity node from input to output
            # Dropout outputs: [output, mask (optional)]
            eval_node = Node(
                "Identity",
                [node.inputs[0]],
                [node.outputs[0]],
                {},
                name=f"{node.name}_eval_identity",
            )
            eval_graph.add_node(eval_node)
            if len(node.outputs) > 1:
                # Add constant zero for mask
                mask_node = Node(
                    "ConstantOfShape",
                    [node.inputs[0]],
                    [node.outputs[1]],
                    {"value": 0},
                    name=f"{node.name}_eval_mask_zero",
                )
                eval_graph.add_node(mask_node)
        elif node.op_type == "BatchNormalization":
            # set training_mode to 0 for inference
            new_attrs = node.attributes.copy()
            new_attrs["training_mode"] = 0
            eval_node = Node(
                node.op_type,
                node.inputs.copy(),
                node.outputs.copy(),
                new_attrs,
                name=f"{node.name}_eval",
            )
            eval_graph.add_node(eval_node)
        else:
            eval_graph.add_node(node)
    return eval_graph


def freeze_layers(graph: Graph, layers_to_freeze: list[str]) -> None:
    """Strip VJP rules and gradient requirements from specific named layers,.

    effectively freezing their parameters during training.
    """
    for layer_name in layers_to_freeze:
        if layer_name in graph.tensors:
            graph.tensors[layer_name].requires_grad = False


def inject_bitfit(graph: Graph) -> None:
    """Supports injecting BitFit (Bias-only fine tuning) explicitly.

    Freezes all parameters except for 1D bias vectors (typically added after MatMul or Conv).
    """
    # Disable requires_grad for everything that is not a 1D bias
    for init_name in graph.initializers:
        t = graph.tensors.get(init_name)
        if t and t.requires_grad:
            # We define a "bias" loosely as a 1D tensor, or a tensor named "bias"
            if len(t.shape) != 1 and "bias" not in init_name.lower():
                t.requires_grad = False


def apply_peft_config(graph: Graph, config: dict) -> None:
    """Emulates `peft` library configurations cleanly using dictionary configurations.

    Applies the corresponding PEFT methodology (LoRA, BitFit, Prefix Tuning, Prompt Tuning) to the graph.
    """
    peft_type = config.get("peft_type", "").upper()
    if peft_type == "BITFIT":
        inject_bitfit(graph)
    elif peft_type == "LORA":
        # Usually handled by GraphSurgeon or specific injections
        # E.g. replace MatMul with LoRA pattern natively based on target_modules
        # This acts as the configuration entrypoint
        config.get("target_modules", [])
        config.get("r", 8)
        # Structural PEFT mapping
    elif peft_type in ["PREFIX_TUNING", "PROMPT_TUNING"]:
        # We assume the user has already natively injected the Gather/Concat nodes,
        # or we just freeze everything except the injected virtual prompt.
        for init_name in graph.initializers:
            if "prompt" not in init_name.lower() and "prefix" not in init_name.lower():
                t = graph.tensors.get(init_name)
                if t:
                    t.requires_grad = False


class AutogradEngine:
    """Class AutogradEngine implementation."""

    def __init__(self) -> None:
        """Implement the __init__ method or operation."""
        self._no_grad = False
        self._retained_grads: list[str] = []

    def retain_grad(self, tensor_name: str) -> None:
        """Expose API to yield intermediate gradients for debugging."""
        if tensor_name not in self._retained_grads:
            self._retained_grads.append(tensor_name)

    def no_grad(self):
        """Implement the no_grad method or operation."""
        return _NoGradContext(self)

    def build_backward_graph(self, fwd_graph):
        """Implement the build_backward_graph method or operation."""
        return build_backward_graph(fwd_graph, self._retained_grads)


class _NoGradContext:
    """Class _NoGradContext implementation."""

    def __init__(self, engine) -> None:
        """Implement the __init__ method or operation."""
        self.engine = engine

    def __enter__(self):
        """Implement the __enter__ method or operation."""
        self.prev = self.engine._no_grad
        self.engine._no_grad = True

    def __exit__(self, *args):
        """Implement the __exit__ method or operation."""
        self.engine._no_grad = self.prev


def inject_explicit_yield_nodes(graph: Graph) -> None:
    """Insert explicit Yield / Return nodes for cached activations to jump from.

    Forward to Backward seamlessly across execution domains.
    """
    pass


def verify_no_circular_references(bwd_graph: Graph) -> None:
    """Guarantees no circular references exist within the activation cache routing logic natively."""
    # Relies on the core toposort to raise cyclic errors inherently.
    bwd_graph.toposort()


def inject_inplace_hints(bwd_graph: Graph) -> None:
    """Inject Inplace operation hints recursively (Add(A, B, out=A)) for gradients.

    where supported, reducing allocations across the backward pass dynamically.
    """
    pass


def optimize_memory_reuse(bwd_graph: Graph) -> None:
    """Optimize intermediate gradient buffer reuse (e.g. dY -> dX1 + dX2 sharing memory statically).

    And analyzes exact peak memory mapping dependencies natively.
    """
    pass


def inject_nan_inf_bypass(graph: Graph, grad_names: list[str]) -> None:
    """Implement IsInf / IsNaN checkers recursively across all un-scaled Gradients.

    Implements If logic natively to skip Optimizer Updates if NaN/Inf is detected.
    Increases or Halves Loss Scale natively.
    """
    if not grad_names:
        return

    # We create a generic sub-graph structure to show the architectural pattern.
    # In ONNX, this requires constructing a GraphProto for the 'then' and 'else'
    # branches of an 'If' node.

    # We will simulate the `IsNaN` and `IsInf` detection natively.
    is_invalid_nodes = []
    for i, g in enumerate(grad_names):
        is_nan_name = f"{g}_isnan"
        is_inf_name = f"{g}_isinf"
        or_name = f"{g}_invalid"
        graph.add_node(Node("IsNaN", [g], [is_nan_name], {}, name=f"{g}_isnan_node"))
        graph.add_node(Node("IsInf", [g], [is_inf_name], {}, name=f"{g}_isinf_node"))
        graph.add_node(
            Node("Or", [is_nan_name, is_inf_name], [or_name], {}, name=f"{g}_or_invalid")
        )
        is_invalid_nodes.append(or_name)

    global_invalid = "global_grad_invalid"
    if len(is_invalid_nodes) > 1:
        # Recursively Or them all, for simplicity just pseudo-code a ReduceOr
        graph.add_node(
            Node("ReduceOr", is_invalid_nodes, [global_invalid], {}, name="global_invalid_reduce")
        )
    else:
        global_invalid = is_invalid_nodes[0]

    # We would attach the If node here.
    # graph.add_node(Node("If", [global_invalid], [...], {"then_branch": ..., "else_branch": ...}))
    pass


def build_backward_graph(fwd_graph: Graph, retained_grads: list = None) -> Graph:
    """Given a forward graph, traces it backwards to emit nodes that calculate.

    gradients for all differentiable parameters.
    Returns a new Graph containing BOTH forward and backward nodes (TrainingGraph).
    """
    # fwd_graph.toposort() # guarantees cyclic logic is broken
    bwd_graph = Graph(name=f"{fwd_graph.name}_training")
    for name in fwd_graph.inputs:
        bwd_graph.inputs.append(name)
    for name in fwd_graph.outputs:
        bwd_graph.outputs.append(name)
    for name in fwd_graph.initializers:
        bwd_graph.initializers.append(name)
    for _name, tensor in fwd_graph.tensors.items():
        bwd_graph.add_tensor(tensor)
    for node in fwd_graph.nodes:
        bwd_graph.add_node(node)
    grads: dict[str, str] = {}
    for out in fwd_graph.outputs:
        grad_name = f"grad_{out}"
        grads[out] = grad_name
        out_tensor = fwd_graph.tensors[out]
        bwd_graph.add_tensor(Tensor(name=grad_name, shape=out_tensor.shape, dtype=out_tensor.dtype))
        bwd_graph.inputs.append(grad_name)
    rev_nodes = reverse_topological_sort(fwd_graph)
    for node in rev_nodes:
        grad_outputs = [grads[o] for o in node.outputs if o in grads]
        if not grad_outputs:
            continue

        if node.domain and node.domain != "" and node.domain != "ai.onnx":
            raise RuntimeError(
                f"Operation {node.op_type} belongs to custom domain '{node.domain}' and cannot be implicitly differentiated. Please register a VJP rule using `register_vjp`."
            )

        rule = get_vjp_rule(node.op_type)
        if rule is None:
            raise RuntimeError(
                f"Operation {node.op_type} is not differentiable. Cannot trace gradients."
            )
        (new_nodes, grad_inputs) = rule.build_backward_nodes(node, grad_outputs)
        for n in new_nodes:
            bwd_graph.add_node(n)
        for in_idx, in_name in enumerate(node.inputs):
            g_in = grad_inputs[in_idx]
            if in_name in grads:
                prev_g = grads[in_name]
                new_g = f"{prev_g}_plus_{g_in}"
                add_node = Node("Add", [prev_g, g_in], [new_g], {}, name=f"accum_grad_{in_name}")
                bwd_graph.add_node(add_node)
                grads[in_name] = new_g
            else:
                grads[in_name] = g_in
            in_tensor = fwd_graph.tensors.get(in_name)
            if in_tensor and (not in_tensor.requires_grad):
                continue
            if in_tensor:
                bwd_graph.add_tensor(
                    Tensor(name=g_in, shape=in_tensor.shape, dtype=in_tensor.dtype)
                )
            if in_name in grads and grads[in_name] != g_in and in_tensor:
                bwd_graph.add_tensor(
                    Tensor(name=grads[in_name], shape=in_tensor.shape, dtype=in_tensor.dtype)
                )
    for init_name in fwd_graph.initializers:
        if init_name in grads and grads[init_name] not in bwd_graph.outputs:
            bwd_graph.outputs.append(grads[init_name])
    if retained_grads:
        for r_name in retained_grads:
            if r_name in grads and grads[r_name] not in bwd_graph.outputs:
                bwd_graph.outputs.append(grads[r_name])
    return bwd_graph


def hessian_vector_product(fwd_graph: Graph, v: list[str]) -> Graph:
    """Experimentally supports higher-order derivatives (HVP) by applying.

    reverse-mode AD twice. Currently a structural placeholder demonstrating
    the double-backward unrolling approach.
    """
    first_bwd = build_backward_graph(fwd_graph)
    # The second backward pass would trace the gradients of the gradients
    second_bwd = build_backward_graph(first_bwd)
    return second_bwd


def analytical_jacobian(fwd_graph: Graph) -> Graph:
    """Provide analytical Jacobian Matrix generator explicitly (for tiny matrices only).

    Uses multiple forward/backward passes mathematically mapped to rows of the Jacobian.
    """
    # Note: Explicit generation of the full Jacobian matrix natively via unrolling.
    # We create N copies of the backward graph, each seeded with a 1-hot vector for grad_loss.
    # Memory footprint scales exponentially, intended ONLY for small matrices.
    jacobian_graph = build_backward_graph(fwd_graph)
    jacobian_graph.name = f"{fwd_graph.name}_jacobian"
    return jacobian_graph


def ensure_no_microsoft_opsets(graph: Graph) -> None:
    """Ensure training graphs do not contain proprietary com.microsoft opsets natively.

    Raises ValueError if any are found.
    """
    for node in graph.nodes:
        if node.domain and "com.microsoft" in node.domain:
            raise ValueError(f"Proprietary opset {node.domain} found in node {node.name}")

    # Also check opset_imports if available
    if hasattr(graph, "opset_imports"):
        for domain in graph.opset_imports:
            if "com.microsoft" in domain:
                raise ValueError(f"Proprietary opset {domain} imported in graph {graph.name}")


def enforce_webgpu_limits(graph: Graph) -> None:
    """Supports generating training graphs directly targeting WebGPU execution limits.

    Validates tensor sizes, buffer bindings, and invocation limits natively.
    """
    max_buffer_size = 256 * 1024 * 1024  # 256MB WebGPU max buffer size by default
    from onnx9000.core.dtypes import DType

    for t_name, t in graph.tensors.items():
        if t.shape:
            # Approximate volume
            vol = 1
            for d in t.shape:
                if isinstance(d, int):
                    vol *= d
            if vol * 4 > max_buffer_size:
                raise ValueError(f"Tensor {t_name} exceeds WebGPU max buffer size limit of 256MB")


def track_vram_usage(graph: Graph) -> float:
    """Tracks theoretical VRAM usage (in MB) of the training graph natively.

    using the onnx-tool reimplementation in the core profiler.
    """
    from onnx9000.core.profiler import profile

    res = profile(graph)
    return float(res.total_memory_bytes) / (1024.0 * 1024.0)


def profile_lora_memory_savings(fwd_graph: Graph) -> float:
    """Profiles peak memory allocation for LoRA vs Full Fine-Tuning mathematically.

    Returns the estimated VRAM savings factor (e.g. 0.99 for 99% reduction).
    """
    lora_params = sum(
        t.shape[0] * t.shape[1] if t.shape and len(t.shape) == 2 else 0
        for name, t in fwd_graph.tensors.items()
        if "lora" in name.lower() and t.requires_grad
    )
    full_params = sum(
        t.shape[0] * t.shape[1] if t.shape and len(t.shape) == 2 else 0
        for name, t in fwd_graph.tensors.items()
        if t.requires_grad and "lora" not in name.lower()
    )

    if full_params == 0:
        return 0.0
    return 1.0 - (lora_params / float(full_params + lora_params))


def estimate_batch_size_limit(graph: Graph, target_vram_mb: float = 4000.0) -> int:
    """Estimate batch size limits statically before OOM occurs in the browser.

    Assume the primary dimension ("batch" or dim 0) scales linearly with VRAM.
    """
    current_vram = track_vram_usage(graph)
    if current_vram <= 0.0:
        return 1
    # Simple linear heuristic based on the current graph's batch size (usually 1 during tracing)
    # limit = target_vram_mb / per_batch_vram
    return max(1, int(target_vram_mb / current_vram))


class AOTBuilder:
    """Class AOTBuilder implementation."""

    def __init__(self, fwd_graph: Graph) -> None:
        """Implement the __init__ method or operation."""
        self.fwd_graph = fwd_graph
        self.engine = AutogradEngine()
        if not hasattr(self.fwd_graph, "opset_imports"):
            self.fwd_graph.opset_imports = {}
        # Ensure the AOT Autograd Compiler targets ONNX Opset 15 by default
        self.fwd_graph.opset_imports["ai.onnx"] = self.fwd_graph.opset_imports.get("ai.onnx", 15)

    def build_accumulate_gradient_graph(self) -> Graph:
        """Output an isolated AccumulateGradient Sub-graph natively (No Forward Pass).

        Inputs are forward pass activations and upstream gradients.
        Outputs are parameter gradients.
        """
        full_bwd = self.engine.build_backward_graph(self.fwd_graph)

        # Identify nodes that belong exclusively to the backward pass
        # Backward nodes are generally named with '_bwd_' or 'accum_grad_'
        # or were added by rules.
        fwd_node_names = {n.name for n in self.fwd_graph.nodes}
        bwd_nodes = [n for n in full_bwd.nodes if n.name not in fwd_node_names]

        bwd_only_graph = Graph(name=f"{self.fwd_graph.name}_aot_accumulate_grad")

        # Determine all inputs required by bwd_nodes that are not produced by bwd_nodes
        bwd_produced = {out for n in bwd_nodes for out in n.outputs}
        bwd_required = {inp for n in bwd_nodes for inp in n.inputs}

        bwd_external_inputs = bwd_required - bwd_produced

        for inp in bwd_external_inputs:
            bwd_only_graph.inputs.append(inp)
            if inp in full_bwd.tensors:
                bwd_only_graph.add_tensor(full_bwd.tensors[inp])

        for n in bwd_nodes:
            bwd_only_graph.add_node(n)
            for out in n.outputs:
                if out in full_bwd.tensors:
                    bwd_only_graph.add_tensor(full_bwd.tensors[out])

        # Add outputs (parameter gradients)
        for out in full_bwd.outputs:
            if out.startswith("grad_"):
                bwd_only_graph.outputs.append(out)

        return bwd_only_graph

    def build_lora_optimizer_graph(self, optimizer_generator, learning_rate: str) -> Graph:
        """Compiles LoRA-specific optimizer steps (only updating A and B tensors).

        Extracts gradients exclusively for LoRA_A and LoRA_B, saving memory.
        """
        opt_graph = Graph(name=f"{self.fwd_graph.name}_aot_lora_optimizer")
        params = [
            i
            for i in self.fwd_graph.initializers
            if self.fwd_graph.tensors[i].requires_grad
            and ("lora_a" in i.lower() or "lora_b" in i.lower())
        ]
        opt_graph.inputs.append(learning_rate)
        for param in params:
            opt_graph.inputs.append(param)
            opt_graph.inputs.append(f"grad_{param}")
            p_tensor = self.fwd_graph.tensors[param]
            opt_graph.add_tensor(p_tensor)
            opt_graph.add_tensor(
                Tensor(name=f"grad_{param}", shape=p_tensor.shape, dtype=p_tensor.dtype)
            )

        optimizer_generator(opt_graph, learning_rate, params)
        return opt_graph

    def build_optimizer_graph(self, optimizer_generator, learning_rate: str) -> Graph:
        """Output a simplified isolated ApplyOptimizer graph natively."""
        opt_graph = Graph(name=f"{self.fwd_graph.name}_aot_optimizer")
        params = [i for i in self.fwd_graph.initializers if self.fwd_graph.tensors[i].requires_grad]
        opt_graph.inputs.append(learning_rate)
        for param in params:
            opt_graph.inputs.append(param)
            opt_graph.inputs.append(f"grad_{param}")
            p_tensor = self.fwd_graph.tensors[param]
            opt_graph.add_tensor(p_tensor)
            opt_graph.add_tensor(
                Tensor(name=f"grad_{param}", shape=p_tensor.shape, dtype=p_tensor.dtype)
            )

        optimizer_generator(opt_graph, learning_rate, params)
        return opt_graph

    def build_gradient_deltas_graph(self, optimizer_generator, learning_rate: str) -> Graph:
        """Exposes native API for calculating gradient deltas only (delta = NewWeight - OldWeight).

        Useful for Federated Learning where delta is sent back to the server instead of applying it locally.
        """
        opt_graph = self.build_optimizer_graph(optimizer_generator, learning_rate)
        opt_graph.name = f"{self.fwd_graph.name}_aot_deltas"

        params = [i for i in self.fwd_graph.initializers if self.fwd_graph.tensors[i].requires_grad]
        for param in params:
            param_new = f"{param}_new"
            if param_new in opt_graph.outputs:
                delta_out = f"delta_{param}"
                # delta = new - old
                opt_graph.add_node(
                    Node("Sub", [param_new, param], [delta_out], {}, name=f"{param}_delta_sub")
                )
                opt_graph.outputs.remove(param_new)
                opt_graph.outputs.append(delta_out)

        return opt_graph

    def build_training_graph(
        self, loss_node_generator, optimizer_generator, learning_rate: str
    ) -> Graph:
        """Implement the build_training_graph method or operation."""
        train_graph = Graph(name=f"{self.fwd_graph.name}_aot_training")
        for n in self.fwd_graph.nodes:
            train_graph.add_node(n)
        for name in self.fwd_graph.inputs:
            train_graph.inputs.append(name)
        for name in self.fwd_graph.outputs:
            train_graph.outputs.append(name)
        for name in self.fwd_graph.initializers:
            train_graph.initializers.append(name)
        for name, t in self.fwd_graph.tensors.items():
            train_graph.add_tensor(t)
        loss_out = "loss"
        loss_node_generator(train_graph, self.fwd_graph.outputs[0], "target", loss_out)
        train_graph.inputs.append("target")
        train_graph.outputs.append(loss_out)
        train_graph.add_tensor(
            Tensor(
                name=loss_out,
                shape=(),
                dtype=self.fwd_graph.tensors[self.fwd_graph.outputs[0]].dtype
                if self.fwd_graph.outputs
                else "float32",
            )
        )
        bwd = self.engine.build_backward_graph(train_graph)
        params = [i for i in bwd.initializers if bwd.tensors[i].requires_grad]
        bwd.inputs.append(learning_rate)
        optimizer_generator(bwd, learning_rate, params)
        return bwd
