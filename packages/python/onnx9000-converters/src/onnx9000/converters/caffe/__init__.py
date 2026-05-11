"""Caffe converter module."""

import os
from typing import Any

from onnx9000.converters.caffe.mapper import CaffeMapper
from onnx9000.converters.caffe.parser import parse_prototxt
from onnx9000.converters.caffe.weights import load_caffemodel
from onnx9000.converters.parsers import BaseParser
from onnx9000.core.ir import Graph


class CaffeConverter(BaseParser):
    """Converter for Caffe models."""

    def __init__(self, weights_path: str):
        """Initialize the converter.

        Args:
            weights_path: Path to the .caffemodel file.
        """
        self.weights_path = weights_path

    def parse(self, model: str) -> Graph:
        """Parse a Caffe .prototxt file and .caffemodel weights into an ONNX9000 Core IR Graph.

        Args:
            model: String content of the .prototxt file or path to .prototxt file.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        if os.path.exists(model):
            with open(model) as f:
                content = f.read()
        else:
            content = model

        net_info = parse_prototxt(content)

        with open(self.weights_path, "rb") as f:
            weights = load_caffemodel(f)

        mapper = CaffeMapper(net_info, weights)
        graph = mapper.map()

        return graph


__all__ = ["parse_prototxt", "load_caffemodel", "CaffeMapper", "CaffeConverter"]
