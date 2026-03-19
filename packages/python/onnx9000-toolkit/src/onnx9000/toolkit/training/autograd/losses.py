"""Loss Functions.

Generators for common loss functions, constructed entirely out of standard ONNX
operations to ensure AOT compilation compatibility.
"""

from onnx9000.core.ir import Graph, Node


def add_mse_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Add MSE loss to the graph."""
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


def add_categorical_crossentropy_loss(
    graph: Graph,
    pred: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> None:
    """Add CategoricalCrossEntropy loss natively to the graph.

    Formula: -Sum(smoothed_target * log(pred)).
    """
    log_pred = f"{loss_out}_log"
    mul_out = f"{loss_out}_mul"
    sum_out = f"{loss_out}_sum_per_batch"
    neg_out = f"{loss_out}_neg"
    graph.add_node(Node("Log", [pred], [log_pred], {}, name=f"{loss_out}_log_node"))

    current_target = target
    if label_smoothing > 0.0:
        smoothed_target = f"{loss_out}_smoothed_target"
        c_smooth = f"{loss_out}_c_smooth"
        c_1_m_smooth = f"{loss_out}_c_1_m_smooth"
        graph.add_node(
            Node("Constant", [], [c_smooth], {"value": [label_smoothing]}, name=f"{loss_out}_c_sm")
        )
        graph.add_node(
            Node(
                "Constant",
                [],
                [c_1_m_smooth],
                {"value": [1.0 - label_smoothing]},
                name=f"{loss_out}_c_1msm",
            )
        )

        # We don't necessarily know num_classes statically, but we can compute it from target shape
        # For simplicity in explicit tracing, assume num_classes is derived or user passes smoothed targets.
        # But wait, label smoothing formula requires num_classes.
        # smoothed_target = target * (1 - label_smoothing) + label_smoothing / num_classes
        # We can dynamically get num_classes using Shape -> Slice -> Cast
        shape_out = f"{loss_out}_shape"
        num_classes_tensor = f"{loss_out}_num_classes_t"
        num_classes_float = f"{loss_out}_num_classes_f"
        smooth_div = f"{loss_out}_smooth_div"
        target_scaled = f"{loss_out}_target_scaled"

        graph.add_node(Node("Shape", [target], [shape_out], {}, name=f"{loss_out}_shape_node"))
        graph.add_node(
            Node(
                "Constant", [], [f"{loss_out}_idx_c"], {"value": [-1]}, name=f"{loss_out}_idx_const"
            )
        )
        graph.add_node(
            Node(
                "Gather",
                [shape_out, f"{loss_out}_idx_c"],
                [num_classes_tensor],
                {"axis": 0},
                name=f"{loss_out}_gather_nc",
            )
        )
        graph.add_node(
            Node(
                "Cast",
                [num_classes_tensor],
                [num_classes_float],
                {"to": 1},
                name=f"{loss_out}_cast_nc",
            )
        )

        graph.add_node(
            Node(
                "Div",
                [c_smooth, num_classes_float],
                [smooth_div],
                {},
                name=f"{loss_out}_div_smooth",
            )
        )
        graph.add_node(
            Node("Mul", [target, c_1_m_smooth], [target_scaled], {}, name=f"{loss_out}_mul_t_1msm")
        )
        graph.add_node(
            Node(
                "Add",
                [target_scaled, smooth_div],
                [smoothed_target],
                {},
                name=f"{loss_out}_add_smooth",
            )
        )
        current_target = smoothed_target

    graph.add_node(
        Node("Mul", [current_target, log_pred], [mul_out], {}, name=f"{loss_out}_mul_node")
    )
    graph.add_node(
        Node(
            "ReduceSum",
            [mul_out],
            [sum_out],
            {"axes": [-1], "keepdims": 0},
            name=f"{loss_out}_sum_classes",
        )
    )
    graph.add_node(Node("Neg", [sum_out], [neg_out], {}, name=f"{loss_out}_neg_node"))

    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [neg_out], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [neg_out], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [neg_out], [loss_out], {}, name=f"{loss_out}_identity"))


def add_crossentropy_loss(
    graph: Graph,
    logits: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> None:
    """Add CrossEntropy loss to the graph."""
    graph.add_node(
        Node(
            "SoftmaxCrossEntropyLoss",
            [logits, target],
            [loss_out],
            {"reduction": reduction, "ignore_index": ignore_index},
            name=f"{loss_out}_smaxce",
        )
    )


def add_bce_with_logits_loss(
    graph: Graph, logits: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Add BCEWithLogits loss to the graph."""
    graph.add_node(
        Node(
            "BCEWithLogitsLoss",
            [logits, target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_bce_logits",
        )
    )


def add_nll_loss(
    graph: Graph, log_probs: str, target: str, loss_out: str, reduction: str = "mean"
) -> None:
    """Add NLLLoss to the graph."""
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
    """Add L1 loss to the graph."""
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
    """Add Huber loss to the graph."""
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
    """Add Cosine Embedding loss to the graph."""
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
    """Add KLDiv loss to the graph."""
    graph.add_node(
        Node(
            "KLDivLoss",
            [pred, target],
            [loss_out],
            {"reduction": reduction},
            name=f"{loss_out}_kldiv",
        )
    )


def add_dice_loss(
    graph: Graph, pred: str, target: str, loss_out: str, reduction: str = "mean", eps: float = 1e-8
) -> None:
    """Add DiceLoss natively to the graph.

    Formula: 1.0 - (2 * Sum(pred * target) + eps) / (Sum(pred) + Sum(target) + eps).
    """
    inter = f"{loss_out}_inter"
    sum_inter = f"{loss_out}_sum_inter"
    two_inter = f"{loss_out}_2_inter"
    two_inter_eps = f"{loss_out}_2_inter_eps"

    sum_p = f"{loss_out}_sum_p"
    sum_t = f"{loss_out}_sum_t"
    sum_pt = f"{loss_out}_sum_pt"
    sum_pt_eps = f"{loss_out}_sum_pt_eps"

    dice_coeff = f"{loss_out}_dice_coeff"
    dice_loss = f"{loss_out}_dice_loss"

    graph.add_node(Node("Mul", [pred, target], [inter], {}, name=f"{loss_out}_inter_mul"))
    graph.add_node(
        Node(
            "ReduceSum",
            [inter],
            [sum_inter],
            {"axes": [-1], "keepdims": 0},
            name=f"{loss_out}_inter_sum",
        )
    )
    graph.add_node(
        Node("Constant", [], [f"{loss_out}_c_2"], {"value": [2.0]}, name=f"{loss_out}_c2")
    )
    graph.add_node(
        Node("Constant", [], [f"{loss_out}_c_eps"], {"value": [eps]}, name=f"{loss_out}_ceps")
    )
    graph.add_node(
        Node("Constant", [], [f"{loss_out}_c_1"], {"value": [1.0]}, name=f"{loss_out}_c1")
    )

    graph.add_node(
        Node("Mul", [sum_inter, f"{loss_out}_c_2"], [two_inter], {}, name=f"{loss_out}_2inter")
    )
    graph.add_node(
        Node(
            "Add",
            [two_inter, f"{loss_out}_c_eps"],
            [two_inter_eps],
            {},
            name=f"{loss_out}_2inter_eps",
        )
    )

    graph.add_node(
        Node(
            "ReduceSum",
            [pred],
            [sum_p],
            {"axes": [-1], "keepdims": 0},
            name=f"{loss_out}_sum_p_node",
        )
    )
    graph.add_node(
        Node(
            "ReduceSum",
            [target],
            [sum_t],
            {"axes": [-1], "keepdims": 0},
            name=f"{loss_out}_sum_t_node",
        )
    )

    graph.add_node(Node("Add", [sum_p, sum_t], [sum_pt], {}, name=f"{loss_out}_add_pt"))
    graph.add_node(
        Node("Add", [sum_pt, f"{loss_out}_c_eps"], [sum_pt_eps], {}, name=f"{loss_out}_add_pt_eps")
    )

    graph.add_node(
        Node("Div", [two_inter_eps, sum_pt_eps], [dice_coeff], {}, name=f"{loss_out}_div")
    )
    graph.add_node(
        Node("Sub", [f"{loss_out}_c_1", dice_coeff], [dice_loss], {}, name=f"{loss_out}_sub_1")
    )

    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [dice_loss], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [dice_loss], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [dice_loss], [loss_out], {}, name=f"{loss_out}_identity"))


def add_focal_loss(
    graph: Graph,
    pred: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> None:
    """Add Focal Loss natively to the graph.

    Formula: -alpha * (1 - p)^gamma * target * log(p) - (1 - alpha) * p^gamma * (1 - target) * log(1 - p).
    """
    # For simplicity, using BCE focal loss formulation.
    # FL(p, y) = -alpha * y * (1-p)^gamma * log(p) - (1-alpha) * (1-y) * p^gamma * log(1-p)
    p = pred
    y = target

    # Constants
    graph.add_node(
        Node("Constant", [], [f"{loss_out}_c1"], {"value": [1.0]}, name=f"{loss_out}_c1_node")
    )
    graph.add_node(
        Node(
            "Constant", [], [f"{loss_out}_alpha"], {"value": [alpha]}, name=f"{loss_out}_alpha_node"
        )
    )
    graph.add_node(
        Node(
            "Constant",
            [],
            [f"{loss_out}_1m_alpha"],
            {"value": [1.0 - alpha]},
            name=f"{loss_out}_1m_alpha_node",
        )
    )
    graph.add_node(
        Node(
            "Constant", [], [f"{loss_out}_gamma"], {"value": [gamma]}, name=f"{loss_out}_gamma_node"
        )
    )

    # 1 - p
    one_m_p = f"{loss_out}_1m_p"
    graph.add_node(Node("Sub", [f"{loss_out}_c1", p], [one_m_p], {}, name=f"{loss_out}_sub_1m_p"))

    # 1 - y
    one_m_y = f"{loss_out}_1m_y"
    graph.add_node(Node("Sub", [f"{loss_out}_c1", y], [one_m_y], {}, name=f"{loss_out}_sub_1m_y"))

    # log(p)
    log_p = f"{loss_out}_log_p"
    graph.add_node(Node("Log", [p], [log_p], {}, name=f"{loss_out}_log_p_node"))

    # log(1-p)
    log_1m_p = f"{loss_out}_log_1m_p"
    graph.add_node(Node("Log", [one_m_p], [log_1m_p], {}, name=f"{loss_out}_log_1m_p_node"))

    # (1-p)^gamma
    one_m_p_gamma = f"{loss_out}_1m_p_gamma"
    graph.add_node(
        Node(
            "Pow",
            [one_m_p, f"{loss_out}_gamma"],
            [one_m_p_gamma],
            {},
            name=f"{loss_out}_pow_1m_p_gamma",
        )
    )

    # p^gamma
    p_gamma = f"{loss_out}_p_gamma"
    graph.add_node(
        Node("Pow", [p, f"{loss_out}_gamma"], [p_gamma], {}, name=f"{loss_out}_pow_p_gamma")
    )

    # Term 1: alpha * y * (1-p)^gamma * log(p)
    t1_1 = f"{loss_out}_t1_1"
    t1_2 = f"{loss_out}_t1_2"
    t1_3 = f"{loss_out}_t1_3"
    graph.add_node(Node("Mul", [f"{loss_out}_alpha", y], [t1_1], {}, name=f"{loss_out}_t1_mul1"))
    graph.add_node(Node("Mul", [t1_1, one_m_p_gamma], [t1_2], {}, name=f"{loss_out}_t1_mul2"))
    graph.add_node(Node("Mul", [t1_2, log_p], [t1_3], {}, name=f"{loss_out}_t1_mul3"))

    # Term 2: (1-alpha) * (1-y) * p^gamma * log(1-p)
    t2_1 = f"{loss_out}_t2_1"
    t2_2 = f"{loss_out}_t2_2"
    t2_3 = f"{loss_out}_t2_3"
    graph.add_node(
        Node("Mul", [f"{loss_out}_1m_alpha", one_m_y], [t2_1], {}, name=f"{loss_out}_t2_mul1")
    )
    graph.add_node(Node("Mul", [t2_1, p_gamma], [t2_2], {}, name=f"{loss_out}_t2_mul2"))
    graph.add_node(Node("Mul", [t2_2, log_1m_p], [t2_3], {}, name=f"{loss_out}_t2_mul3"))

    # FL = -(Term 1 + Term 2)
    fl_sum = f"{loss_out}_fl_sum"
    fl_neg = f"{loss_out}_fl_neg"
    graph.add_node(Node("Add", [t1_3, t2_3], [fl_sum], {}, name=f"{loss_out}_add_t1_t2"))
    graph.add_node(Node("Neg", [fl_sum], [fl_neg], {}, name=f"{loss_out}_neg_fl"))

    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [fl_neg], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [fl_neg], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [fl_neg], [loss_out], {}, name=f"{loss_out}_identity"))


def add_gradient_penalty(
    graph: Graph, grad_names: list[str], loss_out: str, penalty_weight: float = 1.0
) -> None:
    """Supports explicit gradient penalty calculation natively.

    (Norm(dY) - 1.0)**2 added to the original loss.
    Commonly used in WGAN-GP training constraints.
    """
    if not grad_names:
        return

    sq_norms = []
    for g in grad_names:
        sq_g = f"{g}_sq"
        sum_sq_g = f"{g}_sum_sq"
        graph.add_node(Node("Mul", [g, g], [sq_g], {}, name=f"{g}_pen_mul"))
        graph.add_node(Node("ReduceSum", [sq_g], [sum_sq_g], {"keepdims": 0}, name=f"{g}_pen_sum"))
        sq_norms.append(sum_sq_g)

    total_sq_norm = f"{loss_out}_pen_total_sq_norm"
    if len(sq_norms) > 1:
        # Sum(ReduceSum(g^2))
        graph.add_node(Node("Sum", sq_norms, [total_sq_norm], {}, name=f"{loss_out}_pen_sum_all"))
    else:
        total_sq_norm = sq_norms[0]

    norm = f"{loss_out}_pen_norm"
    graph.add_node(Node("Sqrt", [total_sq_norm], [norm], {}, name=f"{loss_out}_pen_sqrt"))

    # (norm - 1.0)^2
    c_1 = f"{loss_out}_pen_c1"
    norm_minus_1 = f"{loss_out}_pen_norm_m1"
    penalty_base = f"{loss_out}_pen_base"

    graph.add_node(Node("Constant", [], [c_1], {"value": [1.0]}, name=f"{loss_out}_pen_c1_node"))
    graph.add_node(Node("Sub", [norm, c_1], [norm_minus_1], {}, name=f"{loss_out}_pen_sub"))
    graph.add_node(
        Node("Mul", [norm_minus_1, norm_minus_1], [penalty_base], {}, name=f"{loss_out}_pen_sq_m1")
    )

    # * penalty_weight
    c_weight = f"{loss_out}_pen_c_weight"
    weighted_penalty = f"{loss_out}_pen_weighted"
    graph.add_node(
        Node(
            "Constant",
            [],
            [c_weight],
            {"value": [penalty_weight]},
            name=f"{loss_out}_pen_c_weight_node",
        )
    )
    graph.add_node(
        Node(
            "Mul",
            [penalty_base, c_weight],
            [weighted_penalty],
            {},
            name=f"{loss_out}_pen_weight_mul",
        )
    )

    # Add to loss
    original_loss = f"{loss_out}_pre_pen"
    for node in graph.nodes:
        if loss_out in node.outputs:
            idx = node.outputs.index(loss_out)
            node.outputs[idx] = original_loss

    graph.add_node(
        Node(
            "Add",
            [original_loss, weighted_penalty],
            [loss_out],
            {},
            name=f"{loss_out}_pen_add_loss",
        )
    )


def add_triplet_margin_loss(
    graph: Graph,
    anchor: str,
    positive: str,
    negative: str,
    loss_out: str,
    reduction: str = "mean",
    margin: float = 1.0,
    p: int = 2,
) -> None:
    """Add TripletMarginLoss to the graph natively."""
    diff_ap = f"{loss_out}_diff_ap"
    diff_an = f"{loss_out}_diff_an"

    graph.add_node(Node("Sub", [anchor, positive], [diff_ap], {}, name=f"{loss_out}_sub_ap"))
    graph.add_node(Node("Sub", [anchor, negative], [diff_an], {}, name=f"{loss_out}_sub_an"))

    d_ap = f"{loss_out}_d_ap"
    d_an = f"{loss_out}_d_an"

    if p == 2:
        sq_ap = f"{loss_out}_sq_ap"
        sq_an = f"{loss_out}_sq_an"
        sum_ap = f"{loss_out}_sum_ap"
        sum_an = f"{loss_out}_sum_an"
        graph.add_node(Node("Mul", [diff_ap, diff_ap], [sq_ap], {}, name=f"{loss_out}_mul_ap"))
        graph.add_node(Node("Mul", [diff_an, diff_an], [sq_an], {}, name=f"{loss_out}_mul_an"))

        graph.add_node(
            Node(
                "ReduceSum",
                [sq_ap],
                [sum_ap],
                {"axes": [-1], "keepdims": 0},
                name=f"{loss_out}_sum_ap_node",
            )
        )
        graph.add_node(
            Node(
                "ReduceSum",
                [sq_an],
                [sum_an],
                {"axes": [-1], "keepdims": 0},
                name=f"{loss_out}_sum_an_node",
            )
        )

        graph.add_node(Node("Sqrt", [sum_ap], [d_ap], {}, name=f"{loss_out}_sqrt_ap"))
        graph.add_node(Node("Sqrt", [sum_an], [d_an], {}, name=f"{loss_out}_sqrt_an"))
    else:
        abs_ap = f"{loss_out}_abs_ap"
        abs_an = f"{loss_out}_abs_an"
        graph.add_node(Node("Abs", [diff_ap], [abs_ap], {}, name=f"{loss_out}_abs_ap_node"))
        graph.add_node(Node("Abs", [diff_an], [abs_an], {}, name=f"{loss_out}_abs_an_node"))
        graph.add_node(
            Node(
                "ReduceSum",
                [abs_ap],
                [d_ap],
                {"axes": [-1], "keepdims": 0},
                name=f"{loss_out}_sum_ap_node",
            )
        )
        graph.add_node(
            Node(
                "ReduceSum",
                [abs_an],
                [d_an],
                {"axes": [-1], "keepdims": 0},
                name=f"{loss_out}_sum_an_node",
            )
        )

    diff_d = f"{loss_out}_diff_d"
    graph.add_node(Node("Sub", [d_ap, d_an], [diff_d], {}, name=f"{loss_out}_sub_d"))

    margin_name = f"{loss_out}_margin"
    graph.add_node(
        Node(
            "ConstantOfShape",
            [d_ap],
            [margin_name],
            {"value": float(margin)},
            name=f"{loss_out}_c_margin",
        )
    )

    add_margin = f"{loss_out}_add_margin"
    graph.add_node(
        Node("Add", [diff_d, margin_name], [add_margin], {}, name=f"{loss_out}_add_margin_node")
    )

    zero_name = f"{loss_out}_zero"
    graph.add_node(
        Node("ConstantOfShape", [d_ap], [zero_name], {"value": 0.0}, name=f"{loss_out}_c_zero")
    )

    max_out = f"{loss_out}_max"
    graph.add_node(Node("Max", [add_margin, zero_name], [max_out], {}, name=f"{loss_out}_max_node"))

    if reduction == "mean":
        graph.add_node(
            Node("ReduceMean", [max_out], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    elif reduction == "sum":
        graph.add_node(
            Node("ReduceSum", [max_out], [loss_out], {"keepdims": 0}, name=f"{loss_out}_reduce")
        )
    else:
        graph.add_node(Node("Identity", [max_out], [loss_out], {}, name=f"{loss_out}_identity"))


def add_margin_ranking_loss(
    graph: Graph,
    input1: str,
    input2: str,
    target: str,
    loss_out: str,
    reduction: str = "mean",
    margin: float = 0.0,
) -> None:
    """Add Margin Ranking loss to the graph."""
    graph.add_node(
        Node(
            "MarginRankingLoss",
            [input1, input2, target],
            [loss_out],
            {"reduction": reduction, "margin": margin},
            name=f"{loss_out}_margin_ranking",
        )
    )
