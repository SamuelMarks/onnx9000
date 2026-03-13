"""
Loss Functions

Provides graph manipulation functions to inject loss calculation
subgraphs (e.g., CrossEntropy, MSE) into the forward graph.
"""

from onnx9000.ir import Graph, Node


def add_mse_loss(
    graph: Graph, pred_name: str, target_name: str, loss_name: str = "loss"
) -> None:
    """Adds a Mean Squared Error loss subgraph."""
    diff_name = f"{loss_name}_diff"
    sq_name = f"{loss_name}_sq"

    node_sub = Node(
        "Sub", [pred_name, target_name], [diff_name], {}, name=f"{loss_name}_sub"
    )
    node_pow = Node(
        "Mul", [diff_name, diff_name], [sq_name], {}, name=f"{loss_name}_pow"
    )
    node_reduce = Node(
        "ReduceMean",
        [sq_name],
        [loss_name],
        {"keepdims": 0},
        name=f"{loss_name}_reduce",
    )

    graph.add_node(node_sub)
    graph.add_node(node_pow)
    graph.add_node(node_reduce)


def add_crossentropy_loss(
    graph: Graph, logits_name: str, labels_name: str, loss_name: str = "loss"
) -> None:
    """Adds a Cross Entropy loss subgraph."""
    sm_name = f"{loss_name}_softmax"
    log_name = f"{loss_name}_log"
    mul_name = f"{loss_name}_mul"

    node_sm = Node(
        "Softmax", [logits_name], [sm_name], {"axis": -1}, name=f"{loss_name}_sm"
    )
    node_log = Node("Log", [sm_name], [log_name], {}, name=f"{loss_name}_log")
    node_mul = Node(
        "Mul", [labels_name, log_name], [mul_name], {}, name=f"{loss_name}_mul"
    )

    neg_name = f"{loss_name}_neg"
    node_neg = Node("Neg", [mul_name], [neg_name], {}, name=f"{loss_name}_neg")
    node_reduce = Node(
        "ReduceMean",
        [neg_name],
        [loss_name],
        {"keepdims": 0},
        name=f"{loss_name}_reduce",
    )

    graph.add_node(node_sm)
    graph.add_node(node_log)
    graph.add_node(node_mul)
    graph.add_node(node_neg)
    graph.add_node(node_reduce)
