from onnx9000.core.ir import Graph
from onnx9000.toolkit.training.autograd.losses import (
    add_bce_with_logits_loss,
    add_cosine_embedding_loss,
    add_crossentropy_loss,
    add_huber_loss,
    add_kldiv_loss,
    add_margin_ranking_loss,
    add_mse_loss,
    add_nll_loss,
)


def test_losses_coverage() -> None:
    g = Graph("g")
    add_mse_loss(g, "p", "t", "l1", reduction="mean")
    add_mse_loss(g, "p", "t", "l2", reduction="sum")
    add_mse_loss(g, "p", "t", "l3", reduction="none")
    add_crossentropy_loss(g, "p", "t", "l4", reduction="mean")
    add_crossentropy_loss(g, "p", "t", "l5", reduction="sum")
    add_crossentropy_loss(g, "p", "t", "l6", reduction="none")
    add_nll_loss(g, "p", "t", "l7", reduction="mean")
    add_nll_loss(g, "p", "t", "l8", reduction="sum")
    add_nll_loss(g, "p", "t", "l9", reduction="none")
    add_huber_loss(g, "p", "t", "l10", delta=1.0, reduction="mean")
    add_huber_loss(g, "p", "t", "l11", delta=1.0, reduction="sum")
    add_huber_loss(g, "p", "t", "l12", delta=1.0, reduction="none")
    add_bce_with_logits_loss(g, "p", "t", "l13", reduction="mean")
    add_kldiv_loss(g, "p", "t", "l14", reduction="mean")
    add_cosine_embedding_loss(g, "p", "t", "margin", "l15", reduction="mean")
    add_margin_ranking_loss(g, "p", "p2", "t", "l16", margin=1.0, reduction="mean")
    assert len(g.nodes) > 10
    from onnx9000.toolkit.training.autograd.losses import add_l1_loss

    add_l1_loss(g, "p", "t", "l17", reduction="mean")
    add_l1_loss(g, "p", "t", "l18", reduction="sum")
    add_l1_loss(g, "p", "t", "l19", reduction="none")
