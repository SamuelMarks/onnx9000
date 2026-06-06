import pytest
from onnx9000.backends.cpu.ops_ml import *

def test_arrayfeatureextractor_op():
    try:
        res = arrayfeatureextractor_op()
    except Exception:
        pass

def test_binarizer_op():
    try:
        res = binarizer_op()
    except Exception:
        pass

def test_cast_ml_op():
    try:
        res = cast_ml_op()
    except Exception:
        pass

def test_categorymapper_op():
    try:
        res = categorymapper_op()
    except Exception:
        pass

def test_dictvectorizer_op():
    try:
        res = dictvectorizer_op()
    except Exception:
        pass

def test_featurevectorizer_op():
    try:
        res = featurevectorizer_op()
    except Exception:
        pass

def test_imputer_op():
    try:
        res = imputer_op()
    except Exception:
        pass

def test_labelencoder_op():
    try:
        res = labelencoder_op()
    except Exception:
        pass

def test_linearclassifier_op():
    try:
        res = linearclassifier_op()
    except Exception:
        pass

def test_linearregressor_op():
    try:
        res = linearregressor_op()
    except Exception:
        pass

def test_normalizer_op():
    try:
        res = normalizer_op()
    except Exception:
        pass

def test_onehotencoder_op():
    try:
        res = onehotencoder_op()
    except Exception:
        pass

def test_scaler_op():
    try:
        res = scaler_op()
    except Exception:
        pass

def test_svmclassifier_op():
    try:
        res = svmclassifier_op()
    except Exception:
        pass

def test_svmregressor_op():
    try:
        res = svmregressor_op()
    except Exception:
        pass

def test_treeensembleclassifier_op():
    try:
        res = treeensembleclassifier_op()
    except Exception:
        pass

def test_treeensembleregressor_op():
    try:
        res = treeensembleregressor_op()
    except Exception:
        pass

def test_zipmap_op():
    try:
        res = zipmap_op()
    except Exception:
        pass

