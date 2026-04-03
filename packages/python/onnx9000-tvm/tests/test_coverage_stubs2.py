import pytest
from onnx9000.tvm.relay.expr import Call, Function, Op, TupleExpr, Var
from onnx9000.tvm.relay.transform.infer_type import TypeChecker
from onnx9000.tvm.relay.ty import FuncType, TensorType


def test_infer_type_call():
    infer = TypeChecker()

    v = Var("x", TensorType((1,), "float32"))
    op = Op("my_op")
    c1 = Call(op, [v])
    assert infer.visit(c1) is not None

    c2 = Call(op, [])
    with pytest.raises(ValueError):
        infer.visit(c2)

    c3 = Call(v, [v])
    with pytest.raises(ValueError):
        infer.visit(c3)

    f = Function([v], v, None, None)
    infer.visit(f)
