from onnx9000.converters.sklearn.builder import SKLearnParser
from onnx9000.core.ir import Graph, Node


class DummyEstimator:
    pass


class DummyPipeline:
    def __init__(self):
        self.steps = [("step1", DummyEstimator())]


def test_pipeline_parse():
    model = DummyPipeline()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert graph is not None
    assert len(graph.nodes) > 0
    assert graph.nodes[0].op == "Identity"


class FeatureUnion:
    def __init__(self):
        self.transformer_list = [("t1", DummyEstimator()), ("t2", DummyEstimator())]


def test_feature_union_parse():
    model = FeatureUnion()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any((n.op == "Concat" for n in graph.nodes))


class ColumnTransformer:
    def __init__(self):
        self.transformers_ = [("t1", DummyEstimator(), [0, 1])]
        self.remainder = "drop"


def test_column_transformer_parse():
    model = ColumnTransformer()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any((n.op == "ArrayFeatureExtractor" for n in graph.nodes))


class DummyScaler:
    def __init__(self):
        import numpy as np

        self.mean_ = np.array([0.0])
        self.scale_ = np.array([1.0])


def test_scaler_parse():
    model = DummyScaler()
    model.__class__.__name__ = "StandardScaler"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any((n.op == "Scaler" for n in graph.nodes))


class DummyLinear:
    def __init__(self):
        import numpy as np

        self.coef_ = np.array([[1.0]])
        self.intercept_ = np.array([0.0])
        self.classes_ = np.array([0, 1])


def test_linear_parse():
    model = DummyLinear()
    model.__class__.__name__ = "LogisticRegression"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any((n.op == "LinearClassifier" for n in graph.nodes))


class DummySearch:
    def __init__(self):
        self.best_estimator_ = DummyLinear()
        self.best_estimator_.__class__.__name__ = "LogisticRegression"


def test_search_parse():
    model = DummySearch()
    model.__class__.__name__ = "GridSearchCV"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any((n.op == "LinearClassifier" for n in graph.nodes))


def test_builder_sklearn_coverage():
    from onnx9000.converters.sklearn.builder import SKLearnParser
    import sys

    # Test _is_type
    parser = SKLearnParser(None)
    assert parser._is_type(1, "int")

    class Pipeline:
        def __init__(self):
            self.steps = [("passthrough", "passthrough"), ("step2", "step2")]

    assert len(parser._parse_estimator(Pipeline(), ["input"])) == 1

    class ColumnTransformer:
        def __init__(self):
            self.transformers_ = [
                ("drop", "drop", [0]),
                ("pass", "passthrough", [1]),
            ]
            self.remainder = "passthrough"

    out = parser._parse_estimator(ColumnTransformer(), ["input"])
    assert len(out) == 1
