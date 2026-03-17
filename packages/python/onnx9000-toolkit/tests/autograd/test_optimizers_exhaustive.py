"""Tests the optimizers exhaustive module functionality."""

import pytest
from onnx9000.core.ir import Graph, Tensor
from onnx9000.toolkit.training.autograd.optimizers import (
    add_adam_optimizer,
    add_differential_privacy_noise,
    add_gradient_clipping_value,
    add_local_dp_gradient_clipping,
    add_rmsprop_optimizer,
)


def test_adam_weight_decay():
    """Tests the adam weight decay functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("p", [2, 2], "float32", requires_grad=True))
    add_adam_optimizer(g, "lr", ["p"], weight_decay=0.01)
    assert any(n.name == "p_wd_c" for n in g.nodes)


def test_rmsprop_momentum_weight_decay():
    """Tests the rmsprop momentum weight decay functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("p", [2, 2], "float32", requires_grad=True))
    add_rmsprop_optimizer(g, "lr", ["p"], momentum=0.9, weight_decay=0.01)
    assert any(n.name == "p_mul_m_mom" for n in g.nodes)


def test_rmsprop_no_momentum():
    """Tests the rmsprop no momentum functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("p", [2, 2], "float32", requires_grad=True))
    add_rmsprop_optimizer(g, "lr", ["p"], momentum=0.0)
    assert any(n.name == "p_div_upd" for n in g.nodes)


def test_differential_privacy_noise():
    """Tests the differential privacy noise functionality."""
    g = Graph("g")
    grad_names = ["grad_p"]
    add_differential_privacy_noise(g, grad_names, noise_multiplier=1.0, max_grad_norm=1.0)
    assert any(n.op_type == "RandomNormalLike" for n in g.nodes)
    assert "grad_p_dp_noisy" in grad_names[0]


def test_local_dp_gradient_clipping():
    """Tests the local dp gradient clipping functionality."""
    g = Graph("g")
    grad_names = ["grad_p"]
    add_local_dp_gradient_clipping(g, grad_names, max_l2_norm=1.0)
    assert any(n.name == "local_dp_c_max_norm" for n in g.nodes)
    assert "grad_p_dp_clipped" in grad_names[0]


def test_gradient_clipping_value():
    """Tests the gradient clipping value functionality."""
    g = Graph("g")
    grad_names = ["grad_p"]
    add_gradient_clipping_value(g, grad_names, clip_value=1.0)
    assert any(n.name == "clip_min_c" for n in g.nodes)
    assert "grad_p_val_clipped" in grad_names[0]


def test_early_returns():
    """Tests the early returns functionality."""
    g = Graph("g")
    add_differential_privacy_noise(g, [], noise_multiplier=1.0, max_grad_norm=1.0)
    add_differential_privacy_noise(g, ["grad_p"], noise_multiplier=0.0, max_grad_norm=1.0)
    add_local_dp_gradient_clipping(g, [], max_l2_norm=1.0)
    add_local_dp_gradient_clipping(g, ["grad_p"], max_l2_norm=0.0)
    add_gradient_clipping_value(g, [], clip_value=1.0)
    add_gradient_clipping_value(g, ["grad_p"], clip_value=0.0)
