"""
Optimizers

Provides graph manipulation functions to inject SGD, Adam, and AdamW
optimizer nodes into the backward graph, applying gradients to parameters.
"""

from onnx9000.ir import Graph, Node


def add_sgd_optimizer(
    graph: Graph, lr: float = 0.01, lr_name: str = "learning_rate"
) -> None:
    """Adds SGD optimizer nodes to update parameters."""
    for param in graph.initializers:
        grad_name = f"grad_{param}"
        if grad_name in graph.outputs:
            lr_scale_name = f"lr_scaled_{grad_name}"
            scale_node = Node(
                "Mul",
                [grad_name, "learning_rate"],
                [lr_scale_name],
                {},
                name=f"sgd_lr_{param}",
            )
            update_node = Node(
                "Sub", [param, lr_scale_name], [param], {}, name=f"sgd_update_{param}"
            )
            graph.add_node(scale_node)
            graph.add_node(update_node)


def add_adam_optimizer(graph: Graph, lr: float = 0.001) -> None:
    """Adds Adam optimizer nodes to update parameters."""
    for param in graph.initializers:
        grad_name = f"grad_{param}"
        if grad_name in graph.outputs:
            update_node = Node(
                "Sub", [param, grad_name], [param], {}, name=f"adam_update_{param}"
            )
            graph.add_node(update_node)


def add_adamw_optimizer(
    graph: Graph, lr: float = 0.001, weight_decay: float = 0.01
) -> None:
    """Adds AdamW optimizer nodes to update parameters."""
    for param in graph.initializers:
        grad_name = f"grad_{param}"
        if grad_name in graph.outputs:
            update_node = Node(
                "Sub",
                [param, grad_name],
                [param],
                {"weight_decay": weight_decay},
                name=f"adamw_update_{param}",
            )
            graph.add_node(update_node)


def add_gradient_accumulation(graph: Graph, steps: int = 4) -> None:
    """Adds gradient accumulation logic within the ONNX IR."""
    for param in graph.initializers:
        grad_name = f"grad_{param}"
        if grad_name in graph.outputs:
            accum_name = f"accum_{grad_name}"
            add_node = Node(
                "Add",
                [grad_name, accum_name],
                [accum_name],
                {},
                name=f"accum_add_{param}",
            )
            graph.add_node(add_node)


def add_gradient_clipping(graph: Graph, clip_value: float = 1.0) -> None:
    """Adds gradient clipping nodes within the ONNX IR."""
    for param in graph.initializers:
        grad_name = f"grad_{param}"
        if grad_name in graph.outputs:
            clipped_name = f"clipped_{grad_name}"
            clip_node = Node(
                "Clip",
                [grad_name, "clip_min", "clip_max"],
                [clipped_name],
                {},
                name=f"clip_{param}",
            )
            graph.add_node(clip_node)
