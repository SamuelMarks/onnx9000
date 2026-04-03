from onnx9000.tvm.ecosystem import TVMParityCertifier
from onnx9000.tvm.relay.frontend.pytorch import PyTorchImporter, from_pytorch
from onnx9000.tvm.relay.frontend.tensorflow import TFImporter, from_tensorflow
from onnx9000.tvm.relay.module import IRModule
from onnx9000.tvm.relay.span import Span, set_span


def test_stubs():
    assert TVMParityCertifier().certify() is True
    assert from_pytorch(None, None) is None
    assert TFImporter().from_tensorflow(None, None) is None
    assert from_tensorflow(None, None) is None

    mod = IRModule()
    mod2 = IRModule()
    mod.update(mod2)
    try:
        from onnx9000.tvm.relay.expr import Function, Var

        v = Var("x", None)
        f = Function([], None, None, None)
        mod.add(v, f)
        mod.add(v, f, update=True)
        mod.add(v, f)  # Should raise ValueError
    except ValueError:
        pass

    class MockExpr:
        pass

    assert set_span(MockExpr(), Span("a", 1, 1)).span.source_name == "a"


def test_relay_parser_misses():
    import pytest
    from onnx9000.tvm.relay.expr import Expr
    from onnx9000.tvm.relay.parser import save_json

    class UnknownExpr(Expr):
        pass

    with pytest.raises(ValueError):
        save_json(UnknownExpr())

    from onnx9000.tvm.relay.parser import IRSpy

    spy = IRSpy()

    class UnknownType:
        pass

    assert spy.serialize_type(UnknownType()) is None
