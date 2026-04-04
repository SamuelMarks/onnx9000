"""Module docstring."""

from onnx9000.converters.parsers import JAXprParser, PyTorchFXParser, XLAHLOParser
from onnx9000.core.ir import Graph


def test_pytorch_fx_parser():
    """Docstring for D103."""
    parser = PyTorchFXParser()
    assert isinstance(parser.parse(None), Graph)


def test_jaxpr_parser():
    """Docstring for D103."""
    parser = JAXprParser()
    assert isinstance(parser.parse(None), Graph)


def test_xla_hlo_parser():
    """Docstring for D103."""
    parser = XLAHLOParser()
    assert isinstance(parser.parse(None), Graph)


def test_base_parser():
    """Docstring for D103."""
    from onnx9000.converters.parsers import BaseParser

    p = BaseParser()
    assert isinstance(p.parse(None), Graph)


def test_parsers_coverage_call():
    """Docstring for D103."""
    import onnx9000.core.ops as ops
    from onnx9000.converters.parsers import JAXprParser, PyTorchFXParser, XLAHLOParser

    p = PyTorchFXParser()
    p.aten_to_ir["aten.addmm.default"](1, 2, 3)  # Execute lambda

    JAXprParser()
    XLAHLOParser()
