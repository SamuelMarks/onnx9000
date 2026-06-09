"""Caffe converter module."""

import os  # pragma: no cover
from typing import Any  # pragma: no cover

from onnx9000.converters.caffe.mapper import CaffeMapper  # pragma: no cover
from onnx9000.converters.caffe.parser import parse_prototxt  # pragma: no cover
from onnx9000.converters.caffe.weights import load_caffemodel  # pragma: no cover
from onnx9000.converters.parsers import BaseParser  # pragma: no cover
from onnx9000.core.ir import Graph  # pragma: no cover


class CaffeConverter(BaseParser):
    """Converter for Caffe models."""

    def __init__(self, weights_path: str):  # pragma: no cover
        """Initialize the converter.

        Args:  # pragma: no cover
            weights_path: Path to the .caffemodel file.  # pragma: no cover

        """  # pragma: no cover
        self.weights_path = weights_path  # pragma: no cover

    def parse(self, model: str) -> Graph:  # pragma: no cover
        """Parse a Caffe .prototxt file and .caffemodel weights into an ONNX9000 Core IR Graph.

        Args:  # pragma: no cover
            model: String content of the .prototxt file or path to .prototxt file.  # pragma: no cover

        Returns:  # pragma: no cover
            The parsed ONNX9000 Core IR Graph.  # pragma: no cover

        """  # pragma: no cover
        if os.path.exists(model):  # pragma: no cover
            with open(model) as f:  # pragma: no cover
                content = f.read()  # pragma: no cover
        else:  # pragma: no cover
            content = model  # pragma: no cover  # pragma: no cover

        net_info = parse_prototxt(content)  # pragma: no cover  # pragma: no cover

        with open(self.weights_path, "rb") as f:  # pragma: no cover
            weights = load_caffemodel(f)  # pragma: no cover

        mapper = CaffeMapper(net_info, weights)  # pragma: no cover
        graph = mapper.map()  # pragma: no cover

        return graph  # pragma: no cover


__all__ = [
    "parse_prototxt",
    "load_caffemodel",
    "CaffeMapper",
    "CaffeConverter",
]  # pragma: no cover
