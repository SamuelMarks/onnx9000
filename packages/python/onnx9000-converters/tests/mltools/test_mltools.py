"""Tests the mltools module functionality."""

import json

from onnx9000.converters.mltools.catboost import parse_catboost_dict, parse_catboost_json
from onnx9000.converters.mltools.coreml import parse_coreml_model
from onnx9000.converters.mltools.h2o import parse_h2o
from onnx9000.converters.mltools.libsvm import parse_libsvm
from onnx9000.converters.mltools.lightgbm import parse_lightgbm_dict, parse_lightgbm_json
from onnx9000.converters.mltools.sparkml import parse_sparkml_pipeline
from onnx9000.converters.mltools.xgboost import parse_xgboost_dict, parse_xgboost_json


def test_catboost() -> None:
    """Tests the catboost functionality."""
    data = {
        "model_info": {"loss_function": "RMSE"},
        "trees": [{"splits": [], "leaf_values": [1.0]}],
        "oblivious_trees": [
            {"splits": [{"split_index": 0, "border": 0.5}], "leaf_values": [1.0, 2.0]}
        ],
        "features_info": {"float_features": [{"feature_id": 0}]},
    }
    g = parse_catboost_dict(data)
    assert g.nodes[0].op_type == "TreeEnsembleRegressor"

    g2 = parse_catboost_json(json.dumps(data))
    assert g2.nodes[0].op_type == "TreeEnsembleRegressor"

    data["model_info"]["loss_function"] = "Logloss"
    g3 = parse_catboost_dict(data)
    assert g3.nodes[0].op_type == "TreeEnsembleClassifier"


def test_lightgbm() -> None:
    """Tests the lightgbm functionality."""
    data = {
        "objective": "regression",
        "tree_info": [
            {
                "tree_index": 0,
                "tree_structure": {
                    "split_feature": 0,
                    "threshold": 0.5,
                    "left_child": {"leaf_value": 1.0},
                    "right_child": {"leaf_value": 2.0},
                },
            }
        ],
    }
    g = parse_lightgbm_dict(data)
    assert g.nodes[0].op_type == "TreeEnsembleRegressor"

    g2 = parse_lightgbm_json(json.dumps(data))
    assert g2.nodes[0].op_type == "TreeEnsembleRegressor"

    data["objective"] = "binary"
    g3 = parse_lightgbm_dict(data)
    assert g3.nodes[0].op_type == "TreeEnsembleClassifier"


def test_xgboost() -> None:
    """Tests the xgboost functionality."""
    data = {
        "learner": {
            "learner_model_param": {"num_class": "0"},
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "model": {
                    "trees": [
                        {
                            "id": 0,
                            "tree_param": {"num_nodes": "3"},
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "split_indices": [0, 0, 0],
                            "split_conditions": [0.5, 1.0, 2.0],
                            "default_left": [1, 0, 0],
                            "base_weights": [0.0, 1.0, 2.0],
                        }
                    ]
                }
            },
        }
    }
    g = parse_xgboost_dict(data)
    assert g.nodes[0].op_type == "TreeEnsembleRegressor"

    g2 = parse_xgboost_json(json.dumps(data))
    assert g2.nodes[0].op_type == "TreeEnsembleRegressor"

    data["learner"]["objective"]["name"] = "binary:logistic"
    g3 = parse_xgboost_dict(data)
    assert g3.nodes[0].op_type == "TreeEnsembleClassifier"


def test_others() -> None:
    """Tests the others functionality."""

    # CoreML
    class MockCoreMLRegressor:
        treeEnsembleRegressor = True

    assert parse_coreml_model(MockCoreMLRegressor()).nodes[-1].op_type == "TreeEnsembleRegressor"

    class MockCoreMLClassifier:
        treeEnsembleClassifier = True

    assert parse_coreml_model(MockCoreMLClassifier()).nodes[-1].op_type == "TreeEnsembleClassifier"

    assert parse_coreml_model(None).nodes[-1].op_type == "Identity"

    # H2O
    assert parse_h2o(None).nodes[-1].op_type == "TreeEnsembleRegressor"
    assert parse_h2o('{"algo": "deeplearning"}').nodes[-1].op_type == "MatMul"
    assert parse_h2o('{"algo": "xgboost"}').nodes[-1].op_type == "TreeEnsembleRegressor"
    assert parse_h2o('{"algo": "unknown"}').nodes[-1].op_type == "Identity"
    assert parse_h2o("{bad json").nodes[-1].op_type == "TreeEnsembleRegressor"

    # LibSVM
    libsvm_text = "svm_type c_svc\n\nkernel_type rbf\nrho -0.2\nSV\n1.5 1:1 2:1\n"
    g_libsvm = parse_libsvm(libsvm_text)
    assert g_libsvm.nodes[-1].op_type == "SVMClassifier"
    assert g_libsvm.nodes[-1].attributes["rho"].value == [-0.2]

    libsvm_text_reg = "svm_type svr\nkernel_type linear\nSV\n"
    g_libsvm_reg = parse_libsvm(libsvm_text_reg)
    assert g_libsvm_reg.nodes[-1].op_type == "SVMRegressor"

    # SparkML
    assert parse_sparkml_pipeline(None).nodes[-1].op_type == "TreeEnsembleRegressor"
    assert (
        parse_sparkml_pipeline('{"class": "RandomForest"}').nodes[-1].op_type
        == "TreeEnsembleRegressor"
    )
    assert (
        parse_sparkml_pipeline(
            '{"class": "org.apache.spark.ml.classification.LogisticRegressionModel"}'
        )
        .nodes[-1]
        .op_type
        == "LinearClassifier"
    )
    assert (
        parse_sparkml_pipeline('{"class": "org.apache.spark.ml.regression.LinearRegressionModel"}')
        .nodes[-1]
        .op_type
        == "LinearRegressor"
    )
    assert parse_sparkml_pipeline("{bad json").nodes[-1].op_type == "TreeEnsembleRegressor"


def test_xgboost_softprob() -> None:
    """Tests the xgboost softprob functionality."""
    data = {
        "learner": {
            "learner_model_param": {"num_class": "0"},
            "objective": {"name": "binary:hinge"},
            "gradient_booster": {"model": {"trees": []}},
        }
    }
    from onnx9000.converters.mltools.xgboost import parse_xgboost_dict

    g = parse_xgboost_dict(data)
    assert g.nodes[0].op_type == "TreeEnsembleClassifier"
