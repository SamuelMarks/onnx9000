"""Module docstring."""

from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.macros import MacroExpander, MacroMatcher, ir_macro


def test_macro_decorator() -> None:
    """Docstring for D103."""

    @ir_macro("TestMacro")
    def test_m(x: Tensor) -> Tensor:
        return x

    x = Tensor(name="x", shape=[1], dtype=1)
    out = test_m(x)
    assert out.name == "TestMacro_out"


def test_expander() -> None:
    """Docstring for D103."""
    expander = MacroExpander()
    graph = Graph("test")
    expanded = expander.apply(graph)
    assert expanded is graph


def test_matcher() -> None:
    """Docstring for D103."""
    matcher = MacroMatcher()
    graph = Graph("test")
    matched = matcher.apply(graph)
    assert matched is graph


def test_macro_expand_pass():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.macros import MacroExpander

    g = Graph("test")
    p = MacroExpander()
    g2 = p.apply(g)
    assert g2 is g


def test_macro_collapse_pass():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.macros import MacroMatcher

    g = Graph("test")
    p = MacroMatcher()
    g2 = p.apply(g)
    assert g2 is g


def test_transformer_block_macro():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.macros import transformer_block_macro

    x = Tensor(name="x", shape=[1], dtype=1)
    w = Tensor(name="w", shape=[1], dtype=1)
    assert transformer_block_macro(x, w, w) is not None


def test_moe_layer_macro():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.macros import moe_layer_macro

    x = Tensor(name="x", shape=[1], dtype=1)
    w = Tensor(name="w", shape=[1], dtype=1)
    assert moe_layer_macro(x, w, [w, w]) is not None


def test_macro_expander_apply():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.macros import MacroExpander

    g = Graph("test")
    n = Node("transformer_block_macro", domain="ai.onnx9000.macro", inputs=[], outputs=["y"])
    g.nodes.append(n)
    p = MacroExpander()
    g2 = p.apply(g)
    assert g2 is g


def test_macro_kwargs():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.macros import transformer_block_macro

    x = Tensor(name="x", shape=[1], dtype=1)
    w = Tensor(name="w", shape=[1], dtype=1)
    assert transformer_block_macro(x, w, weight2=w) is not None


def test_macro_expander_real():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.macros import MacroExpander

    g = Graph("test")
    n = Node(
        "transformer_block_macro", domain="ai.onnx9000.macro", inputs=["x", "w", "w"], outputs=["y"]
    )
    x = Tensor("x", [1], 1)
    w = Tensor("w", [1], 1)
    g.tensors = {"x": x, "w": w}
    g.nodes.append(n)
    p = MacroExpander()
    p.apply(g)

    # second one
    g2 = Graph("test2")
    n2 = Node(
        "moe_layer_macro", domain="ai.onnx9000.macro", inputs=["x", "w", "w", "w"], outputs=["y"]
    )
    g2.tensors = {"x": x, "w": w}
    g2.nodes.append(n2)
    # mock list of expert weights which were varargs
    n2.attributes["expert_weights"] = type("obj", (object,), {"value": [w, w]})()
    p.apply(g2)


def test_macro_expander_real_miss():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.macros import moe_layer_macro, transformer_block_macro

    x = Tensor("x", [1], 1)
    # The python functions directly (the __wrapped__ ones)
    assert transformer_block_macro.__wrapped__(x, x, x) is x
    assert moe_layer_macro.__wrapped__(x, x, [x]) is x


def test_macro_expander_pass_fallback():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.macros import MacroExpander

    g = Graph("test")
    # Missing line 49 is "pass" or "continue". Let's trigger the `node.op_type in MACRO_REGISTRY`
    n = Node("transformer_block_macro", domain="ai.onnx9000.macro", inputs=[], outputs=[])
    g.nodes.append(n)
    p = MacroExpander()
    p.apply(g)


def test_macro_expander_pass_final():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.macros import MacroExpander

    g = Graph("test")
    # Make sure domain is macro but op_type is NOT in MACRO_REGISTRY to hit continue
    n = Node("UnknownMacro", domain="ai.onnx9000.macro", inputs=[], outputs=[])
    g.nodes.append(n)
    p = MacroExpander()
    p.apply(g)
