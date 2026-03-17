import pytest
from unittest.mock import Mock, MagicMock
from onnx9000.optimizer.hummingbird.sklearn_parser import (
    parse_decision_tree_classifier,
    parse_decision_tree_regressor,
    parse_random_forest_classifier,
    extract_n_estimators,
)


def test_parse_decision_tree_classifier():
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


def test_extract_n_estimators():
    mock_rf = Mock()
    mock_rf.n_estimators = 100
    assert extract_n_estimators(mock_rf) == 100

    mock_dt = Mock(spec=[])
    assert extract_n_estimators(mock_dt) == 1


def test_sklearn_parser_all():
    import onnx9000.optimizer.hummingbird.sklearn_parser as sp
    from unittest.mock import MagicMock
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
        estimators_ = [MockTree()]
        estimators_ = [MagicMock(tree_=MockTree())]

    m = MockEst()

    # Test all functions
    sp.parse_decision_tree_classifier(m)
    sp.parse_decision_tree_regressor(m)
    sp.parse_random_forest_classifier(m)
    sp.parse_random_forest_regressor(m)
    sp.parse_gradient_boosting_classifier(m)
    sp.parse_gradient_boosting_regressor(m)
    sp.parse_hist_gradient_boosting_classifier(m)
    sp.parse_hist_gradient_boosting_regressor(m)
    sp.parse_isolation_forest(m)
    sp.parse_ada_boost_classifier(m)
    sp.parse_ada_boost_regressor(m)
    sp.parse_extra_trees_classifier(m)
    sp.parse_extra_trees_regressor(m)
    sp.extract_n_estimators(m)
    sp.handle_predict_proba()
    sp.handle_multi_output_regressors()
    sp.handle_multi_label_classification()
    sp.parse_pipeline()
    sp.extract_classes_and_zipmaps()
    sp.parse_linear_regression()
    sp.parse_logistic_regression()
    sp.parse_ridge_lasso_elasticnet()
    sp.parse_sgd_classifier()
    sp.parse_linear_svc()
    sp.parse_svc_poly()
    sp.parse_svc_rbf()
    sp.parse_svc_sigmoid()
    sp.parse_gaussian_nb()
    sp.parse_multinomial_nb()
    sp.parse_bernoulli_nb()
    sp.parse_mlp_classifier()
    sp.optimize_standard_scaler()
    sp.optimize_binarizer()
    sp.optimize_onehot_encoder()

    # Test no tree path
    class MockNoTree:
        pass

    ab = sp.parse_decision_tree_regressor(MockNoTree())
    assert hasattr(ab, "add_node")

    class MockNoTree2:
        pass

    ab2 = sp.parse_decision_tree_classifier(MockNoTree2())
    assert hasattr(ab2, "add_node")
