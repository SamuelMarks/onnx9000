"""Module providing core logic and structural definitions."""

import onnx9000.toolkit.training.autograd.losses as losses
from onnx9000.core.ir import Graph


def test_mse_loss() -> None:
    """Tests the test_mse_loss functionality."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "ReduceMean" for n in g.nodes)


def test_mse_loss_sum() -> None:
    """Tests the test_mse_loss_sum functionality."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss", reduction="sum")
    assert any(n.op_type == "ReduceSum" for n in g.nodes)


def test_mse_loss_none() -> None:
    """Tests the test_mse_loss_none functionality."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss", reduction="none")
    assert any(n.op_type == "Identity" for n in g.nodes)


def test_crossentropy_loss() -> None:
    """Tests the test_crossentropy_loss functionality."""
    g = Graph("test")
    losses.add_crossentropy_loss(g, "logits", "target", "loss")
    assert any(n.op_type == "SoftmaxCrossEntropyLoss" for n in g.nodes)


def test_bce_with_logits_loss() -> None:
    """Tests the test_bce_with_logits_loss functionality."""
    g = Graph("test")
    losses.add_bce_with_logits_loss(g, "logits", "target", "loss")
    assert any(n.op_type == "BCEWithLogitsLoss" for n in g.nodes)


def test_nll_loss() -> None:
    """Tests the test_nll_loss functionality."""
    g = Graph("test")
    losses.add_nll_loss(g, "log_probs", "target", "loss")
    assert any(n.op_type == "NegativeLogLikelihoodLoss" for n in g.nodes)


def test_l1_loss() -> None:
    """Tests the test_l1_loss functionality."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "ReduceMean" for n in g.nodes)


def test_l1_loss_sum() -> None:
    """Tests the test_l1_loss_sum functionality."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss", reduction="sum")
    assert any(n.op_type == "ReduceSum" for n in g.nodes)


def test_l1_loss_none() -> None:
    """Tests the test_l1_loss_none functionality."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss", reduction="none")
    assert any(n.op_type == "Identity" for n in g.nodes)


def test_huber_loss() -> None:
    """Tests the test_huber_loss functionality."""
    g = Graph("test")
    losses.add_huber_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "HuberLoss" for n in g.nodes)


def test_cosine_embedding_loss() -> None:
    """Tests the test_cosine_embedding_loss functionality."""
    g = Graph("test")
    losses.add_cosine_embedding_loss(g, "in1", "in2", "target", "loss")
    assert any(n.op_type == "CosineEmbeddingLoss" for n in g.nodes)


def test_kldiv_loss() -> None:
    """Tests the test_kldiv_loss functionality."""
    g = Graph("test")
    losses.add_kldiv_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "KLDivLoss" for n in g.nodes)


def test_margin_ranking_loss() -> None:
    """Tests the test_margin_ranking_loss functionality."""
    g = Graph("test")
    losses.add_margin_ranking_loss(g, "in1", "in2", "target", "loss")
    assert any(n.op_type == "MarginRankingLoss" for n in g.nodes)
