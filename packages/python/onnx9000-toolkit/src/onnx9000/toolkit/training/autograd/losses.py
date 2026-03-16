"""
Loss Functions

Generators for common loss functions, constructed entirely out of standard ONNX
operations to ensure AOT compilation compatibility.
"""

from onnx9000.core.ir import Graph, Node


def add_mse_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds MSE loss to the graph."""
    diff = f"{loss_out}_diff"
    sq = f"{loss_out}_sq"
    graph.add_node(Node("Sub", [pred, target], [diff], {}, name=f"{loss_out}_sub"))
    graph.add_node(Node("Mul", [diff, diff], [sq], {}, name=f"{loss_out}_mul"))
    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [sq], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [sq], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [sq], [loss_out], {}, name=f"{loss_out}_identity"))


def add_crossentropy_loss(
    graph: Graph, logits: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds CrossEntropy loss to the graph."""
    graph.add_node(
        Node(
            "SoftmaxCrossEntropyLoss",
            [logits, target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_smaxce",
        )
    )


def add_bce_with_logits_loss(
    graph: Graph, logits: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds BCEWithLogits loss to the graph."""
    graph.add_node(Node("Sigmoid", [logits], [f"{loss_out}_sig"], {}, name=f"{loss_out}_sig_node"))
    graph.add_node(
        Node(
            "BinaryCrossEntropyLoss",
            [f"{loss_out}_sig", target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_bce",
        )
    )


def add_nll_loss(
    graph: Graph, log_probs: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds NLLLoss to the graph."""
    graph.add_node(
        Node(
            "NegativeLogLikelihoodLoss",
            [log_probs, target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_nll",
        )
    )


def add_l1_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds L1 loss to the graph."""
    diff = f"{loss_out}_diff"
    abs_diff = f"{loss_out}_abs"
    graph.add_node(Node("Sub", [pred, target], [diff], {}, name=f"{loss_out}_sub"))
    graph.add_node(Node("Abs", [diff], [abs_diff], {}, name=f"{loss_out}_abs_node"))
    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [abs_diff], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [abs_diff], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [abs_diff], [loss_out], {}, name=f"{loss_out}_identity"))


def add_huber_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean", delta: float = 1.0
) -> None:
    """Adds Huber loss to the graph."""
    graph.add_node(
        Node(
            "HuberLoss",
            [pred, target],
            [loss_out],
            {"reduction": reduction, "delta": delta},
            name=f"{loss_out}_huber",
        )
    )


def add_cosine_embedding_loss(
    graph: Graph,
    input1: str,
    input2: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    margin: float = 0.0,
) -> None:
    """Adds Cosine Embedding loss to the graph."""
    graph.add_node(
        Node(
            "CosineEmbeddingLoss",
            [input1, input2, target],
            [loss_out],
            {"reduction": reduction, "margin": margin},
            name=f"{loss_out}_cos_embed",
        )
    )


def add_kldiv_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Adds KLDiv loss to the graph."""
    graph.add_node(
        Node(
            "KLDivLoss",
            [pred, target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_kldiv",
        )
    )


def add_margin_ranking_loss(
    graph: Graph,
    input1: str,
    input2: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    margin: float = 0.0,
) -> None:
    """Adds Margin Ranking loss to the graph."""
    graph.add_node(
        Node(
            "MarginRankingLoss",
            [input1, input2, target],
            [loss_out],
            {"reduction": reduction, "margin": margin},
            name=f"{loss_out}_margin_ranking",
        )
    )
