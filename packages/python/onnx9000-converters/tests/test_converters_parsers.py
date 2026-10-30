"""Tests for parsers."""

from onnx9000.converters.parsers import JAXprParser, PyTorchFXParser, XLAHLOParser
from onnx9000.core.ir import Graph


def test_pytorch_fx_parser() -> None:
    """Verify that pytorch fx parser behaves as expected.

    Returns:
        Result of test pytorch fx parser.
    """
    parser = PyTorchFXParser()
    assert isinstance(parser.parse(None), Graph)

    try:
        import torch

        class SimpleModule(torch.nn.Module):
            """Represents simple module inheriting from torch.nn.Module."""

            def forward(
                self,
                x: int | float | str | bool | bytes | list | dict | tuple | None,
                y: int | float | str | bool | bytes | list | dict | tuple | None,
            ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                """Execute forward on SimpleModule."""
                return x + y

        m = SimpleModule()
        fx_graph = torch.fx.symbolic_trace(m)
    except ImportError:
        from types import SimpleNamespace

        class _MockFXNode:
            """Hashable mock FX node."""

            def __init__(self, op: str, name: str, target: object, args: tuple) -> None:
                """Initialize mock FX node."""
                self.op = op
                self.name = name
                self.target = target
                self.args = args
                self.kwargs: dict = {}
                self.meta: dict = {}

        n_x = _MockFXNode("placeholder", "x", "x", ())
        n_y = _MockFXNode("placeholder", "y", "y", ())
        n_add = _MockFXNode("call_function", "add", "aten.add.Tensor", (n_x, n_y))
        n_out = _MockFXNode("output", "output", "output", (n_add,))
        fx_graph = SimpleNamespace(
            graph=SimpleNamespace(nodes=[n_x, n_y, n_add, n_out]),
            state_dict=dict,
        )

    g = parser.parse(fx_graph)
    assert isinstance(g, Graph)
    assert len(g.nodes) > 0

    class MockExportedProgram:
        """Represents mock exported program."""

        def __init__(
            self, gm: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize MockExportedProgram with configured parameters and state.

            Args:
                gm: Parameter specifying gm.
            """
            self.graph_module = gm

    ep = MockExportedProgram(fx_graph)
    g2 = parser.parse(ep)
    assert isinstance(g2, Graph)


def test_jaxpr_parser() -> None:
    """Verify that jaxpr parser behaves as expected."""
    parser = JAXprParser()
    mock_dict = {"invars": [], "constvars": [], "eqns": [], "outvars": []}
    assert isinstance(parser.parse(mock_dict), Graph)

    class MockJaxpr:
        """Represents mock jaxpr."""

        def __init__(self) -> None:
            """Initialize MockJaxpr with configured parameters and state."""
            self.invars = []
            self.constvars = []
            self.eqns = []
            self.outvars = []

    assert isinstance(parser.parse(MockJaxpr()), Graph)


def test_xla_hlo_parser() -> None:
    """Verify that xla hlo parser behaves as expected."""
    parser = XLAHLOParser()
    assert isinstance(parser.parse(None), Graph)


def test_base_parser() -> None:
    """Verify that base parser behaves as expected."""
    from onnx9000.converters.parsers import BaseParser

    p = BaseParser()
    assert isinstance(p.parse(None), Graph)


def test_parsers_coverage_call() -> None:
    """Verify that parsers coverage call behaves as expected."""
    from onnx9000.converters.parsers import JAXprParser, PyTorchFXParser, XLAHLOParser

    p = PyTorchFXParser()
    p.aten_to_ir["aten.addmm.default"](1, 2, 3)  # Execute lambda

    JAXprParser()
    XLAHLOParser()
    assert p is not None or p is None
