"""Tests the hummingbird sklearn parser module functionality."""

from unittest.mock import MagicMock, Mock

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.sklearn_parser import (
    extract_classes_and_zipmaps,
    extract_n_estimators,
    handle_multi_label_classification,
    handle_multi_output_regressors,
    handle_predict_proba,
    optimize_binarizer,
    optimize_onehot_encoder,
    optimize_standard_scaler,
    parse_ada_boost_classifier,
    parse_ada_boost_regressor,
    parse_bernoulli_nb,
    parse_decision_tree_classifier,
    parse_decision_tree_regressor,
    parse_extra_trees_classifier,
    parse_extra_trees_regressor,
    parse_gaussian_nb,
    parse_gradient_boosting_classifier,
    parse_gradient_boosting_regressor,
    parse_hist_gradient_boosting_classifier,
    parse_hist_gradient_boosting_regressor,
    parse_isolation_forest,
    parse_linear_regression,
    parse_linear_svc,
    parse_logistic_regression,
    parse_mlp_classifier,
    parse_multinomial_nb,
    parse_pipeline,
    parse_random_forest_classifier,
    parse_random_forest_regressor,
    parse_ridge_lasso_elasticnet,
    parse_sgd_classifier,
    parse_svc_poly,
    parse_svc_rbf,
    parse_svc_sigmoid,
)


def test_parse_decision_tree_classifier() -> None:
    """Tests the parse decision tree classifier functionality."""
    # Mock sklearn tree
    mock_tree = MagicMock()
    mock_tree.node_count = 3
    mock_tree.feature = [0, -2, -2]
    mock_tree.threshold = [1.5, -2.0, -2.0]
    mock_tree.children_left = [1, -1, -1]
    mock_tree.children_right = [2, -1, -1]
    # Mocking numpy array flattening logic
    val0 = MagicMock()
    val0.flatten.return_value = [0.0]
    val1 = MagicMock()
    val1.flatten.return_value = [10.0]
    val2 = MagicMock()
    val2.flatten.return_value = [20.0]
    mock_tree.value = [val0, val1, val2]

    mock_estimator = Mock()
    mock_estimator.tree_ = mock_tree

    abstractions = parse_decision_tree_classifier(mock_estimator)

    assert len(abstractions.features) == 3
    assert abstractions.features[0] == 0
    assert abstractions.thresholds[0] == 1.5
    assert abstractions.left_children[0] == 1
    assert abstractions.values[2] == 20.0


def test_extract_n_estimators() -> None:
    """Tests the extract n estimators functionality."""
    mock_rf = Mock()
    mock_rf.n_estimators = 100
    assert extract_n_estimators(mock_rf) == 100

    mock_dt = Mock(spec=[])
    assert extract_n_estimators(mock_dt) == 1


def test_trees() -> None:
    import numpy as np

    class MockTree:
        node_count = 3
        feature = np.array([0, -2, -2])
        threshold = np.array([0.5, -2.0, -2.0])
        children_left = np.array([1, -1, -1])
        children_right = np.array([2, -1, -1])
        value = np.array([[[1.0]], [[0.0]], [[1.0]]])

    class MockEst:
        tree_ = MockTree()
        estimators_ = [MagicMock(tree_=MockTree())]

    m = MockEst()

    parse_decision_tree_classifier(m)
    parse_decision_tree_regressor(m)
    parse_random_forest_classifier(m)
    parse_random_forest_regressor(m)
    parse_gradient_boosting_classifier(m)
    parse_gradient_boosting_regressor(m)
    parse_hist_gradient_boosting_classifier(m)
    parse_hist_gradient_boosting_regressor(m)
    parse_isolation_forest(m)
    parse_ada_boost_classifier(m)
    parse_ada_boost_regressor(m)
    parse_extra_trees_classifier(m)
    parse_extra_trees_regressor(m)

    class MockNoTree:
        __dummy__ = True

    parse_decision_tree_regressor(MockNoTree())
    parse_decision_tree_classifier(MockNoTree())


def test_handlers():
    g = Graph("test")
    handle_predict_proba(g, "logits")
    assert g.nodes[-1].op_type == "Softmax"

    handle_multi_output_regressors(g, ["o1", "o2"])
    assert g.nodes[-1].op_type == "Concat"

    handle_multi_label_classification(g, "logits")
    assert g.nodes[-1].op_type == "Sigmoid"

    parse_pipeline([])
    extract_classes_and_zipmaps(g, [])


def test_linear():
    assert parse_linear_regression(None).name == "LinearRegression"
    assert parse_logistic_regression(None).nodes[-1].op_type == "Sigmoid"
    assert parse_ridge_lasso_elasticnet(None).name == "LinearRegression"
    assert parse_sgd_classifier(None).name == "LinearRegression"
    assert parse_linear_svc(None).nodes[-1].op_type == "Sign"


def test_svc():
    assert parse_svc_poly(None).name == "SVC_Poly"
    assert parse_svc_rbf(None).name == "SVC_RBF"
    assert parse_svc_sigmoid(None).name == "SVC_Sigmoid"


def test_nb():
    assert parse_gaussian_nb(None).name == "GaussianNB"
    assert parse_multinomial_nb(None).name == "MultinomialNB"
    assert parse_bernoulli_nb(None).name == "BernoulliNB"


def test_mlp():
    assert parse_mlp_classifier(None).name == "MLPClassifier"


def test_scalers():
    assert optimize_standard_scaler(None).nodes[-1].op_type == "Div"
    assert optimize_binarizer(None).nodes[-1].op_type == "Cast"
    assert optimize_onehot_encoder(None).nodes[-1].op_type == "Cast"
