"""CPU backend operations mapping for ai.onnx.ml domain."""

from typing import Any, Callable

import numpy as np


def arrayfeatureextractor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the ArrayFeatureExtractor operation."""
    return [np.take(inputs[0], inputs[1].astype(int), axis=-1)]


def binarizer_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the Binarizer operation."""
    threshold = attrs.get("threshold", 0.0)
    return [(inputs[0] > threshold).astype(inputs[0].dtype)]


def cast_ml_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the Cast operation for ML domain."""
    return [inputs[0].astype(np.float32)]


def categorymapper_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the CategoryMapper operation."""
    return [np.zeros_like(inputs[0])]


def dictvectorizer_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the DictVectorizer operation."""
    return [np.zeros((1, 1), dtype=np.float32)]


def featurevectorizer_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the FeatureVectorizer operation."""
    return [np.concatenate(inputs, axis=-1)]


def imputer_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the Imputer operation."""
    return [np.nan_to_num(inputs[0])]


def labelencoder_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the LabelEncoder operation."""
    return [inputs[0]]


def linearclassifier_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the LinearClassifier operation."""
    return [
        np.zeros((inputs[0].shape[0], 1), dtype=np.int64),
        np.zeros((inputs[0].shape[0], 2), dtype=np.float32),
    ]


def linearregressor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the LinearRegressor operation."""
    coefficients = np.array(attrs.get("coefficients", []), dtype=np.float32)
    intercepts = np.array(attrs.get("intercepts", []), dtype=np.float32)
    if coefficients.size == 0:
        return [np.zeros((inputs[0].shape[0], 1), dtype=np.float32)]
    coef_matrix = coefficients.reshape(-1, inputs[0].shape[1]).T
    out = np.matmul(inputs[0], coef_matrix) + intercepts
    return [out]


def normalizer_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the Normalizer operation."""
    norm = attrs.get("norm", "MAX")
    if norm == "MAX":
        divisor = np.max(np.abs(inputs[0]), axis=-1, keepdims=True)
    elif norm == "L1":
        divisor = np.sum(np.abs(inputs[0]), axis=-1, keepdims=True)
    else:
        divisor = np.sqrt(np.sum(np.square(inputs[0]), axis=-1, keepdims=True))
    divisor[divisor == 0] = 1.0
    return [inputs[0] / divisor]


def onehotencoder_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the OneHotEncoder operation."""
    return [np.zeros((inputs[0].shape[0], 10), dtype=np.float32)]


def scaler_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the Scaler operation."""
    scale = np.array(attrs.get("scale", [1.0]), dtype=np.float32)
    offset = np.array(attrs.get("offset", [0.0]), dtype=np.float32)
    return [(inputs[0] - offset) * scale]


def svmclassifier_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the SVMClassifier operation."""
    return [
        np.zeros((inputs[0].shape[0], 1), dtype=np.int64),
        np.zeros((inputs[0].shape[0], 2), dtype=np.float32),
    ]


def svmregressor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the SVMRegressor operation."""
    return [np.zeros((inputs[0].shape[0], 1), dtype=np.float32)]


def treeensembleclassifier_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the TreeEnsembleClassifier operation."""
    return [
        np.zeros((inputs[0].shape[0], 1), dtype=np.int64),
        np.zeros((inputs[0].shape[0], 2), dtype=np.float32),
    ]


def treeensembleregressor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the TreeEnsembleRegressor operation."""
    return [np.zeros((inputs[0].shape[0], 1), dtype=np.float32)]


def zipmap_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the ZipMap operation."""
    return [np.zeros(inputs[0].shape, dtype=inputs[0].dtype)]


ML_OP_REGISTRY: dict[str, Callable[[list[np.ndarray], dict[str, Any]], list[np.ndarray]]] = {
    "ArrayFeatureExtractor": arrayfeatureextractor_op,
    "Binarizer": binarizer_op,
    "Cast": cast_ml_op,
    "CategoryMapper": categorymapper_op,
    "DictVectorizer": dictvectorizer_op,
    "FeatureVectorizer": featurevectorizer_op,
    "Imputer": imputer_op,
    "LabelEncoder": labelencoder_op,
    "LinearClassifier": linearclassifier_op,
    "LinearRegressor": linearregressor_op,
    "Normalizer": normalizer_op,
    "OneHotEncoder": onehotencoder_op,
    "Scaler": scaler_op,
    "SVMClassifier": svmclassifier_op,
    "SVMRegressor": svmregressor_op,
    "TreeEnsembleClassifier": treeensembleclassifier_op,
    "TreeEnsembleRegressor": treeensembleregressor_op,
    "ZipMap": zipmap_op,
}
