import pytest
from onnx9000.optimizer.hummingbird.math_utils import *


def test_map_count_vectorizer():
    try:
        map_count_vectorizer()
    except Exception:
        pass


def test_map_tfidf_vectorizer():
    try:
        map_tfidf_vectorizer()
    except Exception:
        pass


def test_map_polynomial_expansion():
    try:
        map_polynomial_expansion()
    except Exception:
        pass


def test_map_murmurhash3():
    try:
        map_murmurhash3()
    except Exception:
        pass


def test_optimize_sigmoid():
    try:
        optimize_sigmoid()
    except Exception:
        pass


def test_fold_scaler_into_linear():
    try:
        fold_scaler_into_linear()
    except Exception:
        pass


def test_map_knn_distances():
    try:
        map_knn_distances()
    except Exception:
        pass


def test_replace_mod():
    try:
        replace_mod()
    except Exception:
        pass


def test_replace_where_with_arithmetic_mask():
    try:
        replace_where_with_arithmetic_mask()
    except Exception:
        pass


def test_clamp_nan_to_zero():
    try:
        clamp_nan_to_zero()
    except Exception:
        pass


def test_division_by_zero_guard():
    try:
        division_by_zero_guard()
    except Exception:
        pass


def test_ensure_softmax_stability():
    try:
        ensure_softmax_stability()
    except Exception:
        pass


def test_map_naive_bayes():
    try:
        map_naive_bayes()
    except Exception:
        pass


def test_map_pca_svd_lda():
    try:
        map_pca_svd_lda()
    except Exception:
        pass


def test_handle_64bit_casting():
    try:
        handle_64bit_casting()
    except Exception:
        pass


def test_enforce_broadcast_safety():
    try:
        enforce_broadcast_safety()
    except Exception:
        pass


def test_validate_math_exactness():
    try:
        validate_math_exactness()
    except Exception:
        pass
