"""Darknet converter module."""

import os
from typing import Any

from onnx9000.converters.darknet.mapper import DarknetMapper
from onnx9000.converters.darknet.parser import parse_cfg
from onnx9000.converters.darknet.weights import load_weights
from onnx9000.converters.parsers import BaseParser
from onnx9000.core.ir import Graph


class DarknetConverter(BaseParser):
    """Converter for Darknet models."""

    def __init__(self, weights_path: str):
        """Initialize the converter.

        Args:
            weights_path: Path to the .weights file.
        """
        self.weights_path = weights_path

    def parse(self, model: str) -> Graph:
        """Parse a Darknet .cfg file and weights into an ONNX9000 Core IR Graph.

        Args:
            model: String content of the .cfg file or path to .cfg file.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        # If it's a file path
        if os.path.exists(model):
            with open(model) as f:
                content = f.read()
        else:
            content = model

        layers = parse_cfg(content)

        with open(self.weights_path, "rb") as f:
            weights_data = load_weights(f)

        mapper = DarknetMapper(layers, weights_data["weights"])
        return mapper.map()


__all__ = ["parse_cfg", "load_weights", "DarknetMapper", "DarknetConverter"]
