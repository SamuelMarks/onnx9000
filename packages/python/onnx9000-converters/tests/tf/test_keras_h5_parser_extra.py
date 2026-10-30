"""Test suite for validating the Keras 2 H5 model parser.

This module provides additional test coverage for specific edge cases and gaps in
the Keras H5 parser implementation, such as sequential model sequence resolution,
handling of missing configuration metadata, and dataset weight collection.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser


def test_keras_h5_parser_sequential_gap() -> None:
    """Verify parsing of a Keras sequential model with specific gaps.

    This test checks that Keras sequential models with multiple dense layers
    are parsed correctly, verifying that the previous layer name resolution
    logic is triggered and functions as expected.

    Returns:
        None
    """
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


def test_keras_h5_parser_parse_empty() -> None:
    """Verify parsing behavior when the model configuration is missing from H5 file.

    This test ensures that a ValueError is correctly raised when the parser
    attempts to parse an H5 file that does not contain 'model_config' attributes.

    Returns:
        None
    """
    mock_file = MagicMock()
    mock_file.attrs = {}
    mock_file.__contains__.return_value = False

    with patch("h5py.File", return_value=mock_file):
        parser = KerasH5Parser(filename="dummy.h5")
        with pytest.raises(ValueError, match="Keras model_config not found"):
            parser.parse()


def test_keras_h5_parser_get_weights_dataset() -> None:
    """Verify weight collection when traversing H5 dataset objects.

    This test ensures that the `collect_weights` helper correctly identifies
    dataset objects (triggering h5py.Dataset check) and extracts their values.

    Returns:
        None
    """
    mock_file = MagicMock()
    mock_file.__contains__.side_effect = lambda x: x == "model_weights"

    class MockDataset:
        """Mock dataset representation for simulating h5py.Dataset behavior in tests."""

        def __getitem__(self, val: int | tuple[int, ...]) -> np.ndarray:
            """Retrieve the value of the mock dataset.

            Args:
                val (int | tuple[int, ...]): Key or index to retrieve.

            Returns:
                np.ndarray: Simulated numpy array containing the weight data.
            """
            return np.array([1.0])

    mock_ds = MockDataset()
    mock_lg = MagicMock()

    def mock_visit(func: int | float | str | bool | bytes | list | dict | tuple | None) -> None:
        """Simulate H5 group visit behavior by invoking the visitor callback.

        This function mocks the H5 group `visititems` method, passing a simulated
        weight dataset path and a dataset object to the callback.

        Args:
            func: Callback function to be executed for the visited item.

        Returns:
            None
        """
        # The parser uses a lambda that calls collect_weights(obj, current_weights)
        func("weight_0", mock_ds)

    mock_lg.visititems.side_effect = mock_visit
    mock_weights_group = {"layer_1": mock_lg}
    mock_file.__getitem__.side_effect = lambda x: (
        mock_weights_group if x == "model_weights" else MagicMock()
    )

    # Patch both h5py.File (for __init__) and h5py.Dataset (for isinstance check)
    with patch("h5py.File", return_value=mock_file):
        parser = KerasH5Parser(filename="dummy.h5")
        weights = parser.get_weights()
        assert "layer_1" in weights
        assert len(weights["layer_1"]) == 1
        assert weights["layer_1"][0][0] == 1.0
