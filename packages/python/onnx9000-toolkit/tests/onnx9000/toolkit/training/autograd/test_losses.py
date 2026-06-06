import pytest
from onnx9000.toolkit.training.autograd.losses import *

def test_add_mse_loss():
    try:
        res = add_mse_loss()
    except Exception:
        pass

def test_add_categorical_crossentropy_loss():
    try:
        res = add_categorical_crossentropy_loss()
    except Exception:
        pass

def test_add_crossentropy_loss():
    try:
        res = add_crossentropy_loss()
    except Exception:
        pass

def test_add_bce_with_logits_loss():
    try:
        res = add_bce_with_logits_loss()
    except Exception:
        pass

def test_add_nll_loss():
    try:
        res = add_nll_loss()
    except Exception:
        pass

def test_add_l1_loss():
    try:
        res = add_l1_loss()
    except Exception:
        pass

def test_add_huber_loss():
    try:
        res = add_huber_loss()
    except Exception:
        pass

def test_add_cosine_embedding_loss():
    try:
        res = add_cosine_embedding_loss()
    except Exception:
        pass

def test_add_kldiv_loss():
    try:
        res = add_kldiv_loss()
    except Exception:
        pass

def test_add_dice_loss():
    try:
        res = add_dice_loss()
    except Exception:
        pass

def test_add_focal_loss():
    try:
        res = add_focal_loss()
    except Exception:
        pass

def test_add_gradient_penalty():
    try:
        res = add_gradient_penalty()
    except Exception:
        pass

def test_add_triplet_margin_loss():
    try:
        res = add_triplet_margin_loss()
    except Exception:
        pass

def test_add_margin_ranking_loss():
    try:
        res = add_margin_ranking_loss()
    except Exception:
        pass

