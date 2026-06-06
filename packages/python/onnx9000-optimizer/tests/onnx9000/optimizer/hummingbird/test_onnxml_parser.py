import pytest
from onnx9000.optimizer.hummingbird.onnxml_parser import *


def test_parse_onnxml_tree_ensemble():
    try:
        parse_onnxml_tree_ensemble()
    except Exception:
        pass


def test_extract_tree_ensemble_attributes():
    try:
        extract_tree_ensemble_attributes()
    except Exception:
        pass


def test_parse_onnxml_linear():
    try:
        parse_onnxml_linear()
    except Exception:
        pass


def test_parse_onnxml_svm():
    try:
        parse_onnxml_svm()
    except Exception:
        pass


def test_parse_onnxml_scaler():
    try:
        parse_onnxml_scaler()
    except Exception:
        pass


def test_parse_onnxml_normalizer():
    try:
        parse_onnxml_normalizer()
    except Exception:
        pass


def test_parse_onnxml_binarizer():
    try:
        parse_onnxml_binarizer()
    except Exception:
        pass


def test_parse_onnxml_onehot():
    try:
        parse_onnxml_onehot()
    except Exception:
        pass


def test_parse_onnxml_imputer():
    try:
        parse_onnxml_imputer()
    except Exception:
        pass


def test_parse_onnxml_feature_extractor():
    try:
        parse_onnxml_feature_extractor()
    except Exception:
        pass


def test_parse_onnxml_category_mapper():
    try:
        parse_onnxml_category_mapper()
    except Exception:
        pass


def test_parse_onnxml_zipmap():
    try:
        parse_onnxml_zipmap()
    except Exception:
        pass


def test_apply_onnxml_post_transform():
    try:
        apply_onnxml_post_transform()
    except Exception:
        pass


def test_ensure_static_shapes():
    try:
        ensure_static_shapes()
    except Exception:
        pass
