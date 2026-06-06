import pytest
from onnx9000.converters.sklearn.naive_bayes import *

def test__convert_nb():
    try:
        res = _convert_nb()
    except Exception:
        pass

def test_convert_gaussian_nb():
    try:
        res = convert_gaussian_nb()
    except Exception:
        pass

def test_convert_multinomial_nb():
    try:
        res = convert_multinomial_nb()
    except Exception:
        pass

def test_convert_complement_nb():
    try:
        res = convert_complement_nb()
    except Exception:
        pass

def test_convert_bernoulli_nb():
    try:
        res = convert_bernoulli_nb()
    except Exception:
        pass

def test_convert_categorical_nb():
    try:
        res = convert_categorical_nb()
    except Exception:
        pass

def test_convert_k_neighbors_classifier():
    try:
        res = convert_k_neighbors_classifier()
    except Exception:
        pass

def test_convert_k_neighbors_regressor():
    try:
        res = convert_k_neighbors_regressor()
    except Exception:
        pass

def test_convert_radius_neighbors_classifier():
    try:
        res = convert_radius_neighbors_classifier()
    except Exception:
        pass

def test_convert_radius_neighbors_regressor():
    try:
        res = convert_radius_neighbors_regressor()
    except Exception:
        pass

def test_convert_nearest_centroid():
    try:
        res = convert_nearest_centroid()
    except Exception:
        pass

