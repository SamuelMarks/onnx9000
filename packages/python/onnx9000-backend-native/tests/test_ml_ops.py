import numpy as np
from onnx9000.backends.cpu.ops_ml import ML_OP_REGISTRY


def test_ml_ops():
    fn = ML_OP_REGISTRY.get("ArrayFeatureExtractor")
    res = fn([np.array([[1, 2, 3]]), np.array([1])], {})
    assert res[0].shape == (1, 1)
    fn = ML_OP_REGISTRY.get("Binarizer")
    res = fn([np.array([1.0, -1.0])], {"threshold": 0.0})
    assert res[0].tolist() == [1.0, 0.0]
    fn = ML_OP_REGISTRY.get("Cast")
    res = fn([np.array([1], dtype=np.int32)], {})
    assert res[0].dtype == np.float32
    fn = ML_OP_REGISTRY.get("CategoryMapper")
    res = fn([np.array([1])], {})
    assert res[0].shape == (1,)
    fn = ML_OP_REGISTRY.get("DictVectorizer")
    res = fn([np.array([1])], {})
    assert res[0].shape == (1, 1)
    fn = ML_OP_REGISTRY.get("FeatureVectorizer")
    res = fn([np.array([1]), np.array([2])], {})
    assert res[0].shape == (2,)
    fn = ML_OP_REGISTRY.get("Imputer")
    res = fn([np.array([1.0, np.nan])], {})
    assert res[0].tolist() == [1.0, 0.0]
    fn = ML_OP_REGISTRY.get("LabelEncoder")
    res = fn([np.array([1])], {})
    assert res[0].tolist() == [1]
    fn = ML_OP_REGISTRY.get("LinearClassifier")
    res = fn([np.array([[1.0]])], {})
    assert len(res) == 2
    fn = ML_OP_REGISTRY.get("LinearRegressor")
    res = fn([np.array([[1.0, 2.0]])], {"coefficients": [1.0, 1.0], "intercepts": [0.5]})
    assert res[0].tolist() == [[3.5]]
    fn = ML_OP_REGISTRY.get("Normalizer")
    res = fn([np.array([[1.0, 2.0]])], {"norm": "MAX"})
    assert res[0].tolist() == [[0.5, 1.0]]
    res = fn([np.array([[1.0, 2.0]])], {"norm": "L1"})
    assert res[0].tolist() == [[1.0 / 3.0, 2.0 / 3.0]]
    res = fn([np.array([[3.0, 4.0]])], {"norm": "L2"})
    assert res[0].tolist() == [[0.6, 0.8]]
    fn = ML_OP_REGISTRY.get("OneHotEncoder")
    res = fn([np.array([[1]])], {})
    assert res[0].shape == (1, 10)
    fn = ML_OP_REGISTRY.get("Scaler")
    res = fn([np.array([[1.0]])], {"scale": [2.0], "offset": [0.5]})
    assert res[0].tolist() == [[1.0]]
    fn = ML_OP_REGISTRY.get("SVMClassifier")
    res = fn([np.array([[1.0]])], {})
    assert len(res) == 2
    fn = ML_OP_REGISTRY.get("SVMRegressor")
    res = fn([np.array([[1.0]])], {})
    assert res[0].shape == (1, 1)
    fn = ML_OP_REGISTRY.get("TreeEnsembleClassifier")
    res = fn([np.array([[1.0]])], {})
    assert len(res) == 2
    fn = ML_OP_REGISTRY.get("TreeEnsembleRegressor")
    res = fn([np.array([[1.0]])], {})
    assert res[0].shape == (1, 1)
    fn = ML_OP_REGISTRY.get("ZipMap")
    res = fn([np.array([[1.0]])], {})
    assert res[0].shape == (1, 1)


def test_ml_ops_missing():
    import numpy as np
    from onnx9000.backends.cpu.ops_ml import linearregressor_op

    res = linearregressor_op([np.ones((2, 2), dtype=np.float32)], {"coefficients": []})
    assert res[0].shape == (2, 1)
