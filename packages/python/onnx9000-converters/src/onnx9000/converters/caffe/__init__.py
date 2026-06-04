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
        self.weights_path = weights_path  # pragma: no cover

    def parse(self, model: str) -> Graph:
        """Parse a Caffe .prototxt file and .caffemodel weights into an ONNX9000 Core IR Graph.

        Args:
            model: String content of the .prototxt file or path to .prototxt file.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        if os.path.exists(model):  # pragma: no cover
            with open(model) as f:  # pragma: no cover
                content = f.read()  # pragma: no cover
        else:
            content = model  # pragma: no cover

        net_info = parse_prototxt(content)  # pragma: no cover

        with open(self.weights_path, "rb") as f:  # pragma: no cover
            weights = load_caffemodel(f)  # pragma: no cover

        mapper = CaffeMapper(net_info, weights)  # pragma: no cover
        graph = mapper.map()  # pragma: no cover

        return graph  # pragma: no cover


__all__ = ["parse_prototxt", "load_caffemodel", "CaffeMapper", "CaffeConverter"]
