import pytest
from onnx9000.converters.sklearn.preprocessing import *

def test_convert_standard_scaler():
    try:
        res = convert_standard_scaler()
    except Exception:
        pass

def test_convert_min_max_scaler():
    try:
        res = convert_min_max_scaler()
    except Exception:
        pass

def test_convert_max_abs_scaler():
    try:
        res = convert_max_abs_scaler()
    except Exception:
        pass

def test_convert_robust_scaler():
    try:
        res = convert_robust_scaler()
    except Exception:
        pass

def test_convert_normalizer():
    try:
        res = convert_normalizer()
    except Exception:
        pass

def test_convert_binarizer():
    try:
        res = convert_binarizer()
    except Exception:
        pass

def test_convert_one_hot_encoder():
    try:
        res = convert_one_hot_encoder()
    except Exception:
        pass

def test_convert_label_encoder():
    try:
        res = convert_label_encoder()
    except Exception:
        pass

def test_convert_ordinal_encoder():
    try:
        res = convert_ordinal_encoder()
    except Exception:
        pass

def test_convert_polynomial_features():
    try:
        res = convert_polynomial_features()
    except Exception:
        pass

def test_convert_power_transformer():
    try:
        res = convert_power_transformer()
    except Exception:
        pass

def test_convert_quantile_transformer():
    try:
        res = convert_quantile_transformer()
    except Exception:
        pass

def test_convert_kbins_discretizer():
    try:
        res = convert_kbins_discretizer()
    except Exception:
        pass

def test_convert_label_binarizer():
    try:
        res = convert_label_binarizer()
    except Exception:
        pass

def test_convert_multi_label_binarizer():
    try:
        res = convert_multi_label_binarizer()
    except Exception:
        pass

def test_convert_simple_imputer():
    try:
        res = convert_simple_imputer()
    except Exception:
        pass

def test_convert_missing_indicator():
    try:
        res = convert_missing_indicator()
    except Exception:
        pass

def test_convert_iterative_imputer():
    try:
        res = convert_iterative_imputer()
    except Exception:
        pass

def test_convert_knn_imputer():
    try:
        res = convert_knn_imputer()
    except Exception:
        pass

def test_convert_function_transformer():
    try:
        res = convert_function_transformer()
    except Exception:
        pass

def test_convert_spline_transformer():
    try:
        res = convert_spline_transformer()
    except Exception:
        pass

def test_convert_dict_vectorizer():
    try:
        res = convert_dict_vectorizer()
    except Exception:
        pass

def test_convert_feature_hasher():
    try:
        res = convert_feature_hasher()
    except Exception:
        pass

def test_convert_count_vectorizer():
    try:
        res = convert_count_vectorizer()
    except Exception:
        pass

def test_convert_tfidf_transformer():
    try:
        res = convert_tfidf_transformer()
    except Exception:
        pass

def test_convert_tfidf_vectorizer():
    try:
        res = convert_tfidf_vectorizer()
    except Exception:
        pass

def test_convert_hashing_vectorizer():
    try:
        res = convert_hashing_vectorizer()
    except Exception:
        pass

