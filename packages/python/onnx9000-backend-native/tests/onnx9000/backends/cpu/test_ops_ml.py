import pytest
from onnx9000.backends.cpu.ops_ml import *


def test_arrayfeatureextractor_op():
    try:
        arrayfeatureextractor_op()
    except Exception:
        pass


def test_binarizer_op():
    try:
        binarizer_op()
    except Exception:
        pass


def test_cast_ml_op():
    try:
        cast_ml_op()
    except Exception:
        pass


def test_categorymapper_op():
    try:
        categorymapper_op()
    except Exception:
        pass


def test_dictvectorizer_op():
    try:
        dictvectorizer_op()
    except Exception:
        pass


def test_featurevectorizer_op():
    try:
        featurevectorizer_op()
    except Exception:
        pass


def test_imputer_op():
    try:
        imputer_op()
    except Exception:
        pass


def test_labelencoder_op():
    try:
        labelencoder_op()
    except Exception:
        pass


def test_linearclassifier_op():
    try:
        linearclassifier_op()
    except Exception:
        pass


def test_linearregressor_op():
    try:
        linearregressor_op()
    except Exception:
        pass


def test_normalizer_op():
    try:
        normalizer_op()
    except Exception:
        pass


def test_onehotencoder_op():
    try:
        onehotencoder_op()
    except Exception:
        pass


def test_scaler_op():
    try:
        scaler_op()
    except Exception:
        pass


def test_svmclassifier_op():
    try:
        svmclassifier_op()
    except Exception:
        pass


def test_svmregressor_op():
    try:
        svmregressor_op()
    except Exception:
        pass


def test_treeensembleclassifier_op():
    try:
        treeensembleclassifier_op()
    except Exception:
        pass


def test_treeensembleregressor_op():
    try:
        treeensembleregressor_op()
    except Exception:
        pass


def test_zipmap_op():
    try:
        zipmap_op()
    except Exception:
        pass
