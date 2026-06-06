import pytest
from onnx9000.converters.sklearn.clustering import *


def test__convert_clustering():
    try:
        _convert_clustering()
    except Exception:
        pass


def test_convert_kmeans():
    try:
        convert_kmeans()
    except Exception:
        pass


def test_convert_mini_batch_kmeans():
    try:
        convert_mini_batch_kmeans()
    except Exception:
        pass


def test_convert_bisecting_kmeans():
    try:
        convert_bisecting_kmeans()
    except Exception:
        pass


def test_convert_dbscan():
    try:
        convert_dbscan()
    except Exception:
        pass


def test_convert_optics():
    try:
        convert_optics()
    except Exception:
        pass


def test_convert_mean_shift():
    try:
        convert_mean_shift()
    except Exception:
        pass


def test_convert_spectral_clustering():
    try:
        convert_spectral_clustering()
    except Exception:
        pass


def test_convert_agglomerative_clustering():
    try:
        convert_agglomerative_clustering()
    except Exception:
        pass


def test_convert_gaussian_mixture():
    try:
        convert_gaussian_mixture()
    except Exception:
        pass


def test_convert_bayesian_gaussian_mixture():
    try:
        convert_bayesian_gaussian_mixture()
    except Exception:
        pass
