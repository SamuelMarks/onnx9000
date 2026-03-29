"""Tests the builder new module functionality."""

from onnx9000.converters.sklearn.builder import SKLearnParser


class DummyEstimator:
    """Represents the Dummy Estimator class."""

    __dummy__ = True


class DummyPipeline:
    """Represents the Dummy Pipeline class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.steps = [("step1", DummyEstimator())]


def test_pipeline_parse() -> None:
    """Tests the pipeline parse functionality."""
    model = DummyPipeline()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert graph is not None
    assert len(graph.nodes) > 0
    assert graph.nodes[0].op == "Identity"


class FeatureUnion:
    """Represents the Feature Union class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.transformer_list = [("t1", DummyEstimator()), ("t2", DummyEstimator())]


def test_feature_union_parse() -> None:
    """Tests the feature union parse functionality."""
    model = FeatureUnion()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any(n.op == "Concat" for n in graph.nodes)


class ColumnTransformer:
    """Represents the Column Transformer class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.transformers_ = [("t1", DummyEstimator(), [0, 1])]
        self.remainder = "drop"


def test_column_transformer_parse() -> None:
    """Tests the column transformer parse functionality."""
    model = ColumnTransformer()
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any(n.op == "ArrayFeatureExtractor" for n in graph.nodes)


class DummyScaler:
    """Represents the Dummy Scaler class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        import numpy as np

        self.mean_ = np.array([0.0])
        self.scale_ = np.array([1.0])


def test_scaler_parse() -> None:
    """Tests the scaler parse functionality."""
    model = DummyScaler()
    model.__class__.__name__ = "StandardScaler"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any(n.op == "Scaler" for n in graph.nodes)


class DummyLinear:
    """Represents the Dummy Linear class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        import numpy as np

        self.coef_ = np.array([[1.0]])
        self.intercept_ = np.array([0.0])
        self.classes_ = np.array([0, 1])


def test_linear_parse() -> None:
    """Tests the linear parse functionality."""
    model = DummyLinear()
    model.__class__.__name__ = "LogisticRegression"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any(n.op == "LinearClassifier" for n in graph.nodes)


class DummySearch:
    """Represents the Dummy Search class."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.best_estimator_ = DummyLinear()
        self.best_estimator_.__class__.__name__ = "LogisticRegression"


def test_search_parse() -> None:
    """Tests the search parse functionality."""
    model = DummySearch()
    model.__class__.__name__ = "GridSearchCV"
    parser = SKLearnParser(model)
    graph = parser.parse()
    assert any(n.op == "LinearClassifier" for n in graph.nodes)


def test_builder_sklearn_coverage() -> None:
    """Tests the builder sklearn coverage functionality."""
    from onnx9000.converters.sklearn.builder import SKLearnParser

    # Test _is_type
    parser = SKLearnParser(None)
    assert parser._is_type(1, "int")

    class Pipeline:
        """Represents the Pipeline class and its associated logic."""

        def __init__(self) -> None:
            """Test the init   functionality."""
            self.steps = [("passthrough", "passthrough"), ("step2", "step2")]

    assert len(parser._parse_estimator(Pipeline(), ["input"])) == 1

    class ColumnTransformer:
        """Represents the ColumnTransformer class and its associated logic."""

        def __init__(self) -> None:
            """Test the init   functionality."""
            self.transformers_ = [
                ("drop", "drop", [0]),
                ("pass", "passthrough", [1]),
            ]
            self.remainder = "passthrough"

    out = parser._parse_estimator(ColumnTransformer(), ["input"])
    assert len(out) == 1
