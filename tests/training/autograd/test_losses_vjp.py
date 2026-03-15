"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph
import onnx9000.training.autograd.losses as losses


def test_mse_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "ReduceMean" for n in g.nodes)


def test_mse_loss_sum():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss", reduction="sum")
    assert any(n.op_type == "ReduceSum" for n in g.nodes)


def test_mse_loss_none():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_mse_loss(g, "pred", "target", "loss", reduction="none")
    assert any(n.op_type == "Identity" for n in g.nodes)


def test_crossentropy_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_crossentropy_loss(g, "logits", "target", "loss")
    assert any(n.op_type == "SoftmaxCrossEntropyLoss" for n in g.nodes)


def test_bce_with_logits_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_bce_with_logits_loss(g, "logits", "target", "loss")
    assert any(n.op_type == "BinaryCrossEntropyLoss" for n in g.nodes)


def test_nll_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_nll_loss(g, "log_probs", "target", "loss")
    assert any(n.op_type == "NegativeLogLikelihoodLoss" for n in g.nodes)


def test_l1_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "ReduceMean" for n in g.nodes)


def test_l1_loss_sum():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss", reduction="sum")
    assert any(n.op_type == "ReduceSum" for n in g.nodes)


def test_l1_loss_none():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_l1_loss(g, "pred", "target", "loss", reduction="none")
    assert any(n.op_type == "Identity" for n in g.nodes)


def test_huber_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_huber_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "HuberLoss" for n in g.nodes)


def test_cosine_embedding_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_cosine_embedding_loss(g, "in1", "in2", "target", "loss")
    assert any(n.op_type == "CosineEmbeddingLoss" for n in g.nodes)


def test_kldiv_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_kldiv_loss(g, "pred", "target", "loss")
    assert any(n.op_type == "KLDivLoss" for n in g.nodes)


def test_margin_ranking_loss():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    losses.add_margin_ranking_loss(g, "in1", "in2", "target", "loss")
    assert any(n.op_type == "MarginRankingLoss" for n in g.nodes)
