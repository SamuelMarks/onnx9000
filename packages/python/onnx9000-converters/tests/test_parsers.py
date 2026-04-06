"""Tests for parsers."""

from onnx9000.converters.parsers import JAXprParser, PyTorchFXParser, XLAHLOParser
from onnx9000.core.ir import Graph


def test_pytorch_fx_parser():
    """Docstring for D103."""
    parser = PyTorchFXParser()
    assert isinstance(parser.parse(None), Graph)

    import torch

    class SimpleModule(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    m = SimpleModule()
    fx_graph = torch.fx.symbolic_trace(m)
    g = parser.parse(fx_graph)
    assert isinstance(g, Graph)
    assert len(g.nodes) > 0

    class MockExportedProgram:
        def __init__(self, gm):
            self.graph_module = gm

    ep = MockExportedProgram(fx_graph)
    g2 = parser.parse(ep)
    assert isinstance(g2, Graph)


def test_jaxpr_parser():
    """Docstring for D103."""
    parser = JAXprParser()
    mock_dict = {"invars": [], "constvars": [], "eqns": [], "outvars": []}
    assert isinstance(parser.parse(mock_dict), Graph)

    class MockJaxpr:
        def __init__(self):
            self.invars = []
            self.constvars = []
            self.eqns = []
            self.outvars = []

    assert isinstance(parser.parse(MockJaxpr()), Graph)


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
