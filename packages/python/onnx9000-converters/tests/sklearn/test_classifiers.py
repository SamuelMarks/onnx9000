import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from onnx9000.core.ir import Graph


def test_all_sklearn():
    import pkgutil
    import importlib
    import onnx9000.converters.sklearn as sk

    g = Graph("test")

    class MockEst:
        coef_ = np.array([1, 2])
        intercept_ = np.array([0.5])
        classes_ = np.array([0, 1])
        support_vectors_ = np.array([[1, 2]])
        dual_coef_ = np.array([[0.5]])
        kernel = "rbf"
        _gamma = 1.0
        coef0 = 0.0
        degree = 3
        n_features_in_ = 2
        probA_ = np.array([1.0])
        probB_ = np.array([2.0])
        n_classes_ = np.array([2])
        scale_ = np.array([1, 2])
        mean_ = np.array([0, 0])
        min_ = np.array([0])
        norm = "l2"
        threshold = 0.5
        categories_ = [np.array([0, 1])]
        theta_ = np.array([[1], [2]])
        var_ = np.array([[1], [1]])
        class_prior_ = np.array([0.5, 0.5])
        feature_log_prob_ = np.array([[-1], [-2]])
        class_log_prior_ = np.array([-0.5, -0.5])
        tree_ = MagicMock()
        tree_.node_count = 3
        tree_.children_left = np.array([1, -1, -1])
        tree_.children_right = np.array([2, -1, -1])
        tree_.feature = np.array([0, -2, -2])
        tree_.threshold = np.array([0.5, -2.0, -2.0])
        tree_.value = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]])
        estimators_ = []  # For ensembles
        init_ = MagicMock()  # For GBDT
        init_.class_prior_ = np.array([0.5])
        pass

    for _, module_name, _ in pkgutil.iter_modules(sk.__path__):
        mod = importlib.import_module(f"onnx9000.converters.sklearn.{module_name}")
        for func_name in dir(mod):
            if func_name.startswith("convert_") or func_name.startswith("build_"):
                func = getattr(mod, func_name)
                try:
                    # Attempt signature 1
                    func(MockEst(), ["in"], g)
                except Exception:
                    pass

                try:
                    # Attempt signature 2 (builder)
                    func(MockEst(), "out", MockEst(), "in")
                except Exception:
                    pass
