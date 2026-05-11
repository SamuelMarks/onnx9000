"""MXNet converter module."""

import os

from onnx9000.converters.mxnet.mapper import MXNetMapper
from onnx9000.converters.mxnet.parser import parse_symbol
from onnx9000.converters.mxnet.weights import load_params
from onnx9000.converters.parsers import BaseParser
from onnx9000.core.ir import Graph


class MXNetConverter(BaseParser):
    """Converter for MXNet models."""

    def __init__(self, weights_path: str):
        """Initialize the converter.

        Args:
            weights_path: Path to the .params file.
        """
        self.weights_path = weights_path

    def parse(self, model: str) -> Graph:
        """Parse a MXNet -symbol.json file and .params into an ONNX9000 Core IR Graph.

        Args:
            model: String content of the -symbol.json file or path to it.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        if os.path.exists(model):
            with open(model) as f:
                content = f.read()
        else:
            content = model

        symbol_info = parse_symbol(content)

        with open(self.weights_path, "rb") as f:
            weights = load_params(f)

        mapper = MXNetMapper(symbol_info, weights)
        graph = mapper.map()

        return graph


__all__ = ["parse_symbol", "load_params", "MXNetMapper", "MXNetConverter"]
