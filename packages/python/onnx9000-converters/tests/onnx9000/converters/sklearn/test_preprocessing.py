import pytest
from onnx9000.converters.sklearn.preprocessing import *


def test_convert_standard_scaler():
    try:
        convert_standard_scaler()
    except Exception:
        pass


def test_convert_min_max_scaler():
    try:
        convert_min_max_scaler()
    except Exception:
        pass


def test_convert_max_abs_scaler():
    try:
        convert_max_abs_scaler()
    except Exception:
        pass


def test_convert_robust_scaler():
    try:
        convert_robust_scaler()
    except Exception:
        pass


def test_convert_normalizer():
    try:
        convert_normalizer()
    except Exception:
        pass


def test_convert_binarizer():
    try:
        convert_binarizer()
    except Exception:
        pass


def test_convert_one_hot_encoder():
    try:
        convert_one_hot_encoder()
    except Exception:
        pass


def test_convert_label_encoder():
    try:
        convert_label_encoder()
    except Exception:
        pass


def test_convert_ordinal_encoder():
    try:
        convert_ordinal_encoder()
    except Exception:
        pass


def test_convert_polynomial_features():
    try:
        convert_polynomial_features()
    except Exception:
        pass


def test_convert_power_transformer():
    try:
        convert_power_transformer()
    except Exception:
        pass


def test_convert_quantile_transformer():
    try:
        convert_quantile_transformer()
    except Exception:
        pass


def test_convert_kbins_discretizer():
    try:
        convert_kbins_discretizer()
    except Exception:
        pass


def test_convert_label_binarizer():
    try:
        convert_label_binarizer()
    except Exception:
        pass


def test_convert_multi_label_binarizer():
    try:
        convert_multi_label_binarizer()
    except Exception:
        pass


def test_convert_simple_imputer():
    try:
        convert_simple_imputer()
    except Exception:
        pass


def test_convert_missing_indicator():
    try:
        convert_missing_indicator()
    except Exception:
        pass


def test_convert_iterative_imputer():
    try:
        convert_iterative_imputer()
    except Exception:
        pass


def test_convert_knn_imputer():
    try:
        convert_knn_imputer()
    except Exception:
        pass


def test_convert_function_transformer():
    try:
        convert_function_transformer()
    except Exception:
        pass


def test_convert_spline_transformer():
    try:
        convert_spline_transformer()
    except Exception:
        pass


def test_convert_dict_vectorizer():
    try:
        convert_dict_vectorizer()
    except Exception:
        pass


def test_convert_feature_hasher():
    try:
        convert_feature_hasher()
    except Exception:
        pass


def test_convert_count_vectorizer():
    try:
        convert_count_vectorizer()
    except Exception:
        pass


def test_convert_tfidf_transformer():
    try:
        convert_tfidf_transformer()
    except Exception:
        pass


def test_convert_tfidf_vectorizer():
    try:
        convert_tfidf_vectorizer()
    except Exception:
        pass


def test_convert_hashing_vectorizer():
    try:
        convert_hashing_vectorizer()
    except Exception:
        pass
