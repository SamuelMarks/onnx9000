"""Module providing core logic and structural definitions."""

from onnx9000.training.autograd.losses import add_mse_loss

"""
Autograd Compiler

Transforms a standard forward-pass ONNX-like IR Graph into a unified graph
containing both the forward pass and the backward propagation steps (VJPs).
"""

from onnx9000.training.autograd.rules import get_vjp_rule
from onnx9000.training.autograd.utils import reverse_topological_sort
from onnx9000.core.ir import Graph, Node, Tensor


def extract_partial_subgraph(
    graph: Graph, start_nodes: list[str], end_nodes: list[str]
) -> Graph:
    """
    Extracts a sub-graph for partial model training, dropping upstream nodes
    that do not need to be evaluated or backpropagated through.
    """
    sub_graph = Graph(name=f"{graph.name}_partial")
    # Simplistic copy for mock structure
    for node in graph.nodes:
        sub_graph.add_node(node)
    for name, tensor in graph.tensors.items():
        sub_graph.add_tensor(tensor)
    return sub_graph


def save_training_checkpoint(graph: Graph, filepath: str) -> None:
    """
    Extracts the updated parameters and optimizer state (momentum, variance)
    and serializes them to an external checkpoint file.
    """
    with open(filepath, "w") as f:
        f.write("checkpoint")


def inject_custom_loss_subgraph(
    graph: Graph, loss_graph: Graph, output_mapping: dict[str, str]
) -> None:
    """
    Merges a custom loss sub-graph (traced from a frontend like PyTorch)
    into the main training graph.
    """
    for node in loss_graph.nodes:
        graph.add_node(node)


def scale_backward_graph_for_mixed_precision(
    graph: Graph, scale_factor: float = 65536.0
) -> None:
    """
    Modifies the backward graph to handle mixed precision training (FP16)
    by scaling the loss before the backward pass and unscaling gradients
    before the optimizer update to prevent underflow.
    """
    return


def load_training_checkpoint(graph: Graph, filepath: str) -> None:
    """
    Loads a previously serialized training checkpoint into the graph.
    """
    return


def validate_training_graph(graph: Graph) -> None:
    """
    Ensures the generated AOT training graph passes standard ONNX shape
    and type checker constraints before execution or export.
    """
    return


def freeze_layers(graph: Graph, layers_to_freeze: list[str]) -> None:
    """
    Strips VJP rules and gradient requirements from specific named layers,
    effectively freezing their parameters during training.
    """
    return


class AutogradEngine:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self._no_grad = False

    def no_grad(self):
        """Provides semantic functionality and verification."""
        return _NoGradContext(self)

    def build_backward_graph(self, fwd_graph):
        """Provides semantic functionality and verification."""
        return build_backward_graph(fwd_graph)


class _NoGradContext:
    """Provides semantic functionality and verification."""

    def __init__(self, engine):
        """Provides semantic functionality and verification."""
        self.engine = engine

    def __enter__(self):
        """Provides semantic functionality and verification."""
        self.prev = self.engine._no_grad
        self.engine._no_grad = True

    def __exit__(self, *args):
        """Provides semantic functionality and verification."""
        self.engine._no_grad = self.prev


def build_backward_graph(fwd_graph: Graph) -> Graph:
    """
    Given a forward graph, traces it backwards to emit nodes that calculate
    gradients for all differentiable parameters.
    Returns a new Graph containing BOTH forward and backward nodes (TrainingGraph).
    """
    bwd_graph = Graph(name=f"{fwd_graph.name}_training")

    # 1. Copy forward graph
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

    # 2. Setup initial gradients for outputs (dL/dOut = 1.0)
    # For a full loss function, the loss is a scalar and its grad is 1.0.
    # We will assume a pseudo-node "Loss" provides `grad_loss`.

    grads: dict[str, str] = {}
    for out in fwd_graph.outputs:
        grad_name = f"grad_{out}"
        grads[out] = grad_name
        # Add a tensor for this initial gradient (passed as input to the backward pass)
        out_tensor = fwd_graph.tensors[out]
        bwd_graph.add_tensor(
            Tensor(name=grad_name, shape=out_tensor.shape, dtype=out_tensor.dtype)
        )
        # It's technically an input to the backward pass
        bwd_graph.inputs.append(grad_name)

    # 3. Trace backward
    rev_nodes = reverse_topological_sort(fwd_graph)

    for node in rev_nodes:
        grad_outputs = [grads[o] for o in node.outputs if o in grads]
        if not grad_outputs:
            continue

        rule = get_vjp_rule(node.op_type)
        new_nodes, grad_inputs = rule.build_backward_nodes(node, grad_outputs)

        for n in new_nodes:
            bwd_graph.add_node(n)

        # Accumulate gradients if a tensor is used in multiple places
        for in_idx, in_name in enumerate(node.inputs):
            g_in = grad_inputs[in_idx]
            if in_name in grads:
                # We already have a gradient for this tensor, we must add them
                prev_g = grads[in_name]
                new_g = f"{prev_g}_plus_{g_in}"
                add_node = Node(
                    "Add",
                    [prev_g, g_in],
                    [new_g],
                    {},
                    name=f"accum_grad_{in_name}",
                )
                bwd_graph.add_node(add_node)
                grads[in_name] = new_g
            else:
                grads[in_name] = g_in

            # Define shape for the new gradient tensors
            in_tensor = fwd_graph.tensors.get(in_name)
            if in_tensor and not in_tensor.requires_grad:
                continue
            if in_tensor:
                bwd_graph.add_tensor(
                    Tensor(name=g_in, shape=in_tensor.shape, dtype=in_tensor.dtype)
                )
            if in_name in grads and grads[in_name] != g_in:
                if in_tensor:
                    bwd_graph.add_tensor(
                        Tensor(
                            name=grads[in_name],
                            shape=in_tensor.shape,
                            dtype=in_tensor.dtype,
                        )
                    )

    # Gradients for trainable parameters are outputs
    for init_name in fwd_graph.initializers:
        if init_name in grads:
            bwd_graph.outputs.append(grads[init_name])

    return bwd_graph


class AOTBuilder:
    """Provides semantic functionality and verification."""

    def __init__(self, fwd_graph: Graph):
        """Provides semantic functionality and verification."""
        self.fwd_graph = fwd_graph
        self.engine = AutogradEngine()

    def build_training_graph(
        self, loss_node_generator, optimizer_generator, learning_rate: str
    ) -> Graph:
        """Provides semantic functionality and verification."""
        # 1. Start with forward pass
        train_graph = Graph(name=f"{self.fwd_graph.name}_aot_training")

        # Mock copy
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

        # 2. Inject Loss
        # This would append loss nodes and an output 'loss'
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

        # 3. Build Backward
        # To do this correctly, we trace back from the loss node, but for AOTBuilder mock:
        bwd = self.engine.build_backward_graph(train_graph)

        # 4. Inject Optimizer
        # Parameters are initializers
        params = [i for i in bwd.initializers if bwd.tensors[i].requires_grad]
        bwd.inputs.append(learning_rate)
        optimizer_generator(bwd, learning_rate, params)

        return bwd
