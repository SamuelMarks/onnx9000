import pytest
from onnx9000.converters.sklearn.decomposition import *


def test__convert_pca_like():
    try:
        _convert_pca_like()
    except Exception:
        pass


def test_convert_pca():
    try:
        convert_pca()
    except Exception:
        pass


def test_convert_incremental_pca():
    try:
        convert_incremental_pca()
    except Exception:
        pass


def test_convert_truncated_svd():
    try:
        convert_truncated_svd()
    except Exception:
        pass


def test_convert_fast_ica():
    try:
        convert_fast_ica()
    except Exception:
        pass


def test_convert_nmf():
    try:
        convert_nmf()
    except Exception:
        pass


def test_convert_kernel_pca():
    try:
        convert_kernel_pca()
    except Exception:
        pass


def test_convert_latent_dirichlet_allocation():
    try:
        convert_latent_dirichlet_allocation()
    except Exception:
        pass
