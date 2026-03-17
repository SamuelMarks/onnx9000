import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.math_utils import (
    map_count_vectorizer,
    map_tfidf_vectorizer,
    map_polynomial_expansion,
    map_murmurhash3,
    optimize_sigmoid,
    fold_scaler_into_linear,
    map_knn_distances,
    replace_mod,
    replace_where_with_arithmetic_mask,
    clamp_nan_to_zero,
    division_by_zero_guard,
    ensure_softmax_stability,
    map_naive_bayes,
    map_pca_svd_lda,
    handle_64bit_casting,
    enforce_broadcast_safety,
    validate_math_exactness,
)
from onnx9000.optimizer.hummingbird.perfect_tree import (
    handle_perfect_multi_output,
    map_categorical_perfect,
)


def test_hummingbird_math_stubs():
    g = Graph("g")
    map_count_vectorizer(g, "in", ["a"])
    map_tfidf_vectorizer(g, "in", ["a"], [1.0])
    map_polynomial_expansion(g, "in", 2)
    map_murmurhash3(g, "in")
    optimize_sigmoid(g, "in", "out")
    optimize_sigmoid(g, "in", "out", use_fast_math=True)
    fold_scaler_into_linear(None, None, None, None)
    map_knn_distances(g)
    replace_mod(g, "a", "b", "out")
    replace_where_with_arithmetic_mask(g, "cond", "t", "f", "out")
    clamp_nan_to_zero(g, "in", "out")
    division_by_zero_guard(g, "denom", "eps", "out")
    ensure_softmax_stability(g, "in", "out")
    map_naive_bayes(g)
    map_pca_svd_lda(g)
    handle_64bit_casting(g)
    enforce_broadcast_safety(g)
    validate_math_exactness()
    handle_perfect_multi_output(g)
    map_categorical_perfect(g)
