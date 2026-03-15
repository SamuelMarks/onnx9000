import pytest
from onnx9000.script import script, op, Var

GLOBAL_VAR = 3.14


def test_script_decorator_basic():

    @script
    def my_model(x, y):
        z = op.Add(x, y)
        w = op.Relu(z)
        return w

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    assert len(builder.inputs) == 2
    assert len(builder.outputs) == 1
    assert len(builder.nodes) == 2
    assert builder.nodes[0].op_type == "Add"
    assert builder.nodes[1].op_type == "Relu"


def test_script_decorator_binop():

    @script
    def my_model(x, y):
        a = x + y
        b = a - y
        c = b * x
        d = c / y
        e = d**x
        f = e @ y
        return f

    builder = my_model.to_builder()
    assert len(builder.nodes) == 6
    ops = [n.op_type for n in builder.nodes]
    assert ops == ["Add", "Sub", "Mul", "Div", "Pow", "MatMul"]


def test_script_decorator_closure():
    local_val = 42

    @script
    def my_model(x):
        a = op.Add(x, local_val)
        b = op.Mul(a, GLOBAL_VAR)
        return b

    builder = my_model.to_builder()
    assert len(builder.nodes) == 4
    ops = [n.op_type for n in builder.nodes]
    assert ops.count("Constant") == 2


def test_script_decorator_multi_assign():

    @script
    def my_model(x):
        val, idx = op.TopK(x, 5)
        return val, idx

    builder = my_model.to_builder()
    assert len(builder.nodes) == 2
    assert len(builder.outputs) == 2


def test_script_decorator_if():

    @script
    def my_model(x):
        if x > 0:
            y = x
        else:
            y = op.Neg(x)
        return y

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    assert len(builder.nodes) == 3
    ops = [n.op_type for n in builder.nodes]
    assert "If" in ops
    if_node = next(n for n in builder.nodes if n.op_type == "If")
    assert "then_branch" in if_node.attributes
    assert "else_branch" in if_node.attributes


def test_script_decorator_for():

    @script
    def my_model(x, max_trip):
        res = x
        for i in max_trip:
            res = res + x
        return res

    builder = my_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Loop" in ops


def test_script_decorator_inlining():

    @script
    def inner_model(x):
        return op.Relu(x)

    @script
    def outer_model(x):
        y = op.Add(x, 1)
        z = inner_model(y)
        return z

    builder = outer_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Relu" in ops
    assert "Add" in ops


def test_script_decorator_parse_error():

    @script
    def my_model(x):
        raise RuntimeError("Oops")

    with pytest.raises(ValueError, match="Parse error at line"):
        my_model.to_builder()

    @script
    def my_model(x):
        res = x
        while res < 10:
            res = res + x
        return res

    builder = my_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Loop" in ops


def test_script_decorator_listcomp():

    @script
    def my_model(x):
        res = [i for i in x]
        return res

    with pytest.raises(ValueError, match="List comprehensions cannot be mapped"):
        my_model.to_builder()


def test_script_decorator_annotation_and_empty_return():

    @script
    def my_model(x: "Float[10, 20]", y):
        z = op.Add(x, y)
        return

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    assert len(builder.outputs) == 0


def test_script_decorator_unsupported():

    @script
    def my_model(x):
        print(x)
        return x

    with pytest.raises(ValueError, match="Unsupported call"):
        my_model.to_builder()


def test_script_decorator_unsupported_binop():

    @script
    def my_model(x, y):
        z = x % y
        return z

    with pytest.raises(ValueError, match="Unsupported binary operator"):
        my_model.to_builder()


def test_script_parser_docstring():
    from onnx9000.script import script

    @script
    def dummy_func_with_doc():
        """This is a dummy docstring"""
        return 1


def test_script_parser_docstring2():
    from onnx9000.script.parser import ScriptParser
    import ast

    tree = ast.parse('def func():\n    """hello"""\n    pass')
    parser = ScriptParser(None)
    parser.visit(tree.body[0])
