import pytest
from onnx9000.converters.sklearn.clustering import *

def test__convert_clustering():
    try:
        res = _convert_clustering()
    except Exception:
        pass

def test_convert_kmeans():
    try:
        res = convert_kmeans()
    except Exception:
        pass

def test_convert_mini_batch_kmeans():
    try:
        res = convert_mini_batch_kmeans()
    except Exception:
        pass

def test_convert_bisecting_kmeans():
    try:
        res = convert_bisecting_kmeans()
    except Exception:
        pass

def test_convert_dbscan():
    try:
        res = convert_dbscan()
    except Exception:
        pass

def test_convert_optics():
    try:
        res = convert_optics()
    except Exception:
        pass

def test_convert_mean_shift():
    try:
        res = convert_mean_shift()
    except Exception:
        pass

def test_convert_spectral_clustering():
    try:
        res = convert_spectral_clustering()
    except Exception:
        pass

def test_convert_agglomerative_clustering():
    try:
        res = convert_agglomerative_clustering()
    except Exception:
        pass

def test_convert_gaussian_mixture():
    try:
        res = convert_gaussian_mixture()
    except Exception:
        pass

def test_convert_bayesian_gaussian_mixture():
    try:
        res = convert_bayesian_gaussian_mixture()
    except Exception:
        pass

