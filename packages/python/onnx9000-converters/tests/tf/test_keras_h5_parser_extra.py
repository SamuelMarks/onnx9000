"""Test coverage for Keras 2 H5 model parser - additional gaps."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser


def test_keras_h5_parser_sequential_gap():
    """Verify parsing of a Keras sequential model with specific gaps (lines 120-121)."""
    mock_file = MagicMock()
    # Sequential model config with multiple layers to trigger prev_layer_name logic
    model_config = {
        "class_name": "Sequential",
        "config": {
            "layers": [
                {"class_name": "Dense", "config": {"name": "d1", "units": 10}},
                {"class_name": "Dense", "config": {"name": "d2", "units": 5}},
            ]
        },
    }
    mock_file.attrs = {"model_config": json.dumps(model_config).encode("utf-8")}
    mock_file.__contains__.side_effect = lambda x: x == "model_weights"
    mock_file["model_weights"] = {}

    with patch("h5py.File", return_value=mock_file):
        parser = KerasH5Parser(filename="dummy.h5")
        graph = parser.parse()
        assert len(graph.nodes) == 2
        assert graph.nodes[0].op == "Dense"
        assert graph.nodes[1].op == "Dense"


def test_keras_h5_parser_parse_empty():
    """Verify parsing when model_config is missing."""
    mock_file = MagicMock()
    mock_file.attrs = {}
    mock_file.__contains__.return_value = False

    with patch("h5py.File", return_value=mock_file):
        parser = KerasH5Parser(filename="dummy.h5")
        with pytest.raises(ValueError, match="Keras model_config not found"):
            parser.parse()


def test_keras_h5_parser_get_weights_dataset():
    """Verify collect_weights hits Dataset check (line 129)."""
    import h5py
    import onnx9000.converters.tf.keras_h5_parser as parser_mod

    mock_file = MagicMock()
    mock_file.__contains__.side_effect = lambda x: x == "model_weights"

    class MockDataset:
        """Mock dataset."""

        def __getitem__(self, val):
            """Getitem."""
            return np.array([1.0])

    mock_ds = MockDataset()

    mock_lg = MagicMock()

    def mock_visit(func):
        # The parser uses a lambda that calls collect_weights(obj, current_weights)
        """Mock visit."""
        func("weight_0", mock_ds)

    mock_lg.visititems.side_effect = mock_visit
    mock_weights_group = {"layer_1": mock_lg}
    mock_file.__getitem__.side_effect = lambda x: (
        mock_weights_group if x == "model_weights" else MagicMock()
    )

    # Patch both h5py.File (for __init__) and h5py.Dataset (for isinstance check)
    with patch("h5py.File", return_value=mock_file):
        with patch("onnx9000.converters.tf.keras_h5_parser.h5py.Dataset", MockDataset):
            parser = KerasH5Parser(filename="dummy.h5")
            weights = parser.get_weights()
            assert "layer_1" in weights
            assert len(weights["layer_1"]) == 1
            assert weights["layer_1"][0][0] == 1.0
