"""NCNN converter module."""

import os
from typing import Any

from onnx9000.converters.ncnn.mapper import NCNNMapper
from onnx9000.converters.ncnn.parser import parse_param
from onnx9000.converters.ncnn.weights import WeightsReader
from onnx9000.converters.parsers import BaseParser
from onnx9000.core.ir import Graph


class NCNNConverter(BaseParser):
    """Converter for NCNN models."""

    def __init__(self, weights_path: str):
        """Initialize the converter.

        Args:
            weights_path: Path to the .bin file.
        """
        self.weights_path = weights_path

    def parse(self, model: str) -> Graph:
        """Parse a NCNN .param file and .bin weights into an ONNX9000 Core IR Graph.

        Args:
            model: String content of the .param file or path to .param file.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        if os.path.exists(model):
            with open(model) as f:
                content = f.read()
        else:
            content = model

        param_info = parse_param(content)

        with open(self.weights_path, "rb") as f:
            weights_reader = WeightsReader(f)
            mapper = NCNNMapper(param_info, weights_reader)
            graph = mapper.map()

        return graph


__all__ = ["parse_param", "WeightsReader", "NCNNMapper", "NCNNConverter"]
