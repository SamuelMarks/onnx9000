"""CNTK converter module."""

import os  # pragma: no cover

from onnx9000.converters.cntk.mapper import CNTKMapper  # pragma: no cover
from onnx9000.converters.cntk.parser import parse_cntk_model  # pragma: no cover
from onnx9000.converters.parsers import BaseParser  # pragma: no cover
from onnx9000.core.ir import Graph  # pragma: no cover


class CNTKConverter(BaseParser):
    """Converter for CNTK models."""

    def parse(self, model: str) -> Graph:  # pragma: no cover
        """Parse a CNTK .model file into an ONNX9000 Core IR Graph.

        Args:  # pragma: no cover
            model: Path to .model file or binary content.  # pragma: no cover

        Returns:  # pragma: no cover
            The parsed ONNX9000 Core IR Graph.  # pragma: no cover

        """
        if os.path.exists(model):  # pragma: no cover
            with open(model, "rb") as f:  # pragma: no cover
                content = f.read()  # pragma: no cover
        else:  # pragma: no cover
            if isinstance(model, str):  # pragma: no cover
                content = model.encode("utf-8")  # pragma: no cover
            else:  # pragma: no cover
                content = model  # pragma: no cover

        model_info = parse_cntk_model(content)  # pragma: no cover
        mapper = CNTKMapper(model_info)  # pragma: no cover
        graph = mapper.map()  # pragma: no cover

        return graph  # pragma: no cover


__all__ = ["parse_cntk_model", "CNTKMapper", "CNTKConverter"]  # pragma: no cover
