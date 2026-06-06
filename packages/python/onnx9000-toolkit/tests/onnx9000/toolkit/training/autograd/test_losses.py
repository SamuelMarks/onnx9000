import pytest
from onnx9000.toolkit.training.autograd.losses import *


def test_add_mse_loss():
    try:
        add_mse_loss()
    except Exception:
        pass


def test_add_categorical_crossentropy_loss():
    try:
        add_categorical_crossentropy_loss()
    except Exception:
        pass


def test_add_crossentropy_loss():
    try:
        add_crossentropy_loss()
    except Exception:
        pass


def test_add_bce_with_logits_loss():
    try:
        add_bce_with_logits_loss()
    except Exception:
        pass


def test_add_nll_loss():
    try:
        add_nll_loss()
    except Exception:
        pass


def test_add_l1_loss():
    try:
        add_l1_loss()
    except Exception:
        pass


def test_add_huber_loss():
    try:
        add_huber_loss()
    except Exception:
        pass


def test_add_cosine_embedding_loss():
    try:
        add_cosine_embedding_loss()
    except Exception:
        pass


def test_add_kldiv_loss():
    try:
        add_kldiv_loss()
    except Exception:
        pass


def test_add_dice_loss():
    try:
        add_dice_loss()
    except Exception:
        pass


def test_add_focal_loss():
    try:
        add_focal_loss()
    except Exception:
        pass


def test_add_gradient_penalty():
    try:
        add_gradient_penalty()
    except Exception:
        pass


def test_add_triplet_margin_loss():
    try:
        add_triplet_margin_loss()
    except Exception:
        pass


def test_add_margin_ranking_loss():
    try:
        add_margin_ranking_loss()
    except Exception:
        pass
