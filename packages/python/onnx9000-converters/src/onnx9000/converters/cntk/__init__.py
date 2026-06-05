"""CNTK converter module."""

import os

from onnx9000.converters.cntk.mapper import CNTKMapper
from onnx9000.converters.cntk.parser import parse_cntk_model
from onnx9000.converters.parsers import BaseParser
from onnx9000.core.ir import Graph


class CNTKConverter(BaseParser):
    """Converter for CNTK models."""

    def parse(self, model: str) -> Graph:
        """Parse a CNTK .model file into an ONNX9000 Core IR Graph.

        Args:
            model: Path to .model file or binary content.

        Returns:
            The parsed ONNX9000 Core IR Graph.
        """
        if os.path.exists(model):
            with open(model, "rb") as f:
                content = f.read()
        else:
            if isinstance(model, str):
                content = model.encode("utf-8")
            else:
                content = model

        model_info = parse_cntk_model(content)
        mapper = CNTKMapper(model_info)
        graph = mapper.map()

        return graph


__all__ = ["parse_cntk_model", "CNTKMapper", "CNTKConverter"]
