"""Darknet converter module."""

import os  # pragma: no cover
from typing import Any  # pragma: no cover

from onnx9000.converters.darknet.mapper import DarknetMapper  # pragma: no cover
from onnx9000.converters.darknet.parser import parse_cfg  # pragma: no cover
from onnx9000.converters.darknet.weights import load_weights  # pragma: no cover
from onnx9000.converters.parsers import BaseParser  # pragma: no cover
from onnx9000.core.ir import Graph  # pragma: no cover


class DarknetConverter(BaseParser):
    """Converter for Darknet models."""

    def __init__(self, weights_path: str):
        """Initialize the converter.

        Args:  # pragma: no cover
            weights_path: Path to the .weights file.  # pragma: no cover

        """
        self.weights_path = weights_path  # pragma: no cover

    def parse(self, model: str) -> Graph:
        """Parse a Darknet .cfg file and weights into an ONNX9000 Core IR Graph.

        Args:  # pragma: no cover
            model: String content of the .cfg file or path to .cfg file.  # pragma: no cover

        Returns:  # pragma: no cover
            The parsed ONNX9000 Core IR Graph.  # pragma: no cover

        """
        # If it's a file path  # pragma: no cover
        if os.path.exists(model):  # pragma: no cover
            with open(model) as f:  # pragma: no cover
                content = f.read()  # pragma: no cover
        else:  # pragma: no cover
            content = model  # pragma: no cover

        layers = parse_cfg(content)  # pragma: no cover

        with open(self.weights_path, "rb") as f:  # pragma: no cover
            weights_data = load_weights(f)  # pragma: no cover

        mapper = DarknetMapper(layers, weights_data["weights"])  # pragma: no cover
        return mapper.map()  # pragma: no cover


__all__ = [
    "parse_cfg",
    "load_weights",
    "DarknetMapper",
    "DarknetConverter",
]  # pragma: no cover
