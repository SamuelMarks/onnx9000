"""Tests the losses exhaustive module functionality."""

from onnx9000.core.ir import Graph, Node
from onnx9000.toolkit.training.autograd.losses import (
    add_categorical_crossentropy_loss,
    add_dice_loss,
    add_focal_loss,
    add_gradient_penalty,
    add_triplet_margin_loss,
)


def test_add_categorical_crossentropy_loss():
    """Tests the add categorical crossentropy loss functionality."""
    for red in ["none", "mean", "sum"]:
        g = Graph("g")
        add_categorical_crossentropy_loss(g, "p", "t", "out", reduction=red, label_smoothing=0.0)
        assert len(g.nodes) > 0
        g = Graph("g")
        add_categorical_crossentropy_loss(g, "p", "t", "out", reduction=red, label_smoothing=0.1)
        assert len(g.nodes) > 0


def test_add_dice_loss():
    """Tests the add dice loss functionality."""
    for red in ["none", "mean", "sum"]:
        g = Graph("g")
        add_dice_loss(g, "p", "t", "out", reduction=red)
        assert len(g.nodes) > 0


def test_add_focal_loss():
    """Tests the add focal loss functionality."""
    for red in ["none", "mean", "sum"]:
        for alpha in [0.0, 0.25]:
            g = Graph("g")
            add_focal_loss(g, "p", "t", "out", alpha=alpha, gamma=2.0, reduction=red)
            assert len(g.nodes) > 0


def test_add_gradient_penalty():
    """Tests the add gradient penalty functionality."""
    # Empty grad_names
    g = Graph("g")
    add_gradient_penalty(g, [], "out")

    # One grad_name, no existing loss_out node
    g = Graph("g")
    add_gradient_penalty(g, ["grad_1"], "out")
    assert len(g.nodes) > 0

    # Two grad_names, existing loss_out node
    g = Graph("g")
    g.add_node(Node("Dummy", ["in"], ["out"], {}))
    add_gradient_penalty(g, ["grad_1", "grad_2"], "out")
    assert len(g.nodes) > 0


def test_add_triplet_margin_loss():
    """Tests the add triplet margin loss functionality."""
    for red in ["none", "mean", "sum"]:
        for p in [1, 2]:
            g = Graph("g")
            add_triplet_margin_loss(g, "anc", "pos", "neg", "out", margin=1.0, p=p, reduction=red)
            assert len(g.nodes) > 0
