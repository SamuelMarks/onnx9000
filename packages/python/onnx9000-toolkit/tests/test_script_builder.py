"""Unit tests for the onnx9000 script builder functionality.

This module validates script parser generation, ast transformations, and
the initialization/execution of custom builders.
"""


def test_builder_all() -> None:
    """Verifies that GraphBuilder initializes and adds nodes and inputs properly."""
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.var import Var

    builder = GraphBuilder("test_builder_graph")
    v_in = builder.add_input("x")
    assert isinstance(v_in, Var)
    assert builder.name == "test_builder_graph"


def test_script_parser_full() -> None:
    """Execute unit and integration tests for the ScriptParser.

    This function defines and runs a series of parses using the ScriptParser to ensure
    that complex AST structures, control flows, and valid/invalid constructs are
    handled correctly.

    Returns:
        None
    """
    import pytest
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op
    from onnx9000.toolkit.script.parser import ScriptParser

    # Needs a real function with lots of AST branches
    def full_func(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Parse a valid Python function with a rich variety of operators and AST structures.

        This function tests mapping of tuple assignments, binary operators (+, -, *, /, @, **),
        comparisons (>, <, ==, !=), and direct ONNX operator calls (op.Add).

        Args:
            a (Var): The input variable or tensor to be processed.

        Returns:
            tuple[Var, Var]: A tuple containing the addition result and the original input.
        """
        # AssignTuple
        x, y = a, a
        # BinOp
        b = a + 1
        c = a - 1
        d = a * 1
        e = a / 1
        f = a @ a
        g = a**2
        # Compare
        if x > 1:
            pass
        if y < 1:
            pass
        if c == d:
            pass
        if e != f:
            pass
        # Call op directly
        res = op.Add(g, b)
        # Call global
        # Constants and name resolution covered
        return res, a

    parser = ScriptParser({"op": op})
    b = parser.parse(full_func)
    assert isinstance(b, GraphBuilder)

    def invalid_binop(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test standard floor division which is an unsupported binary operator.

        Args:
            a (Var): The input variable or tensor.

        Returns:
            Var: The floor division result (will trigger parser failure).
        """
        return a // 2  # Floor div not mapped

    with pytest.raises(ValueError):
        parser.parse(invalid_binop)

    def invalid_cmp(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test a comparison containing an unsupported GtE (greater-than-or-equal-to) operator.

        Args:
            a (Var): The input variable.

        Returns:
            bool: Comparison result (will trigger parser failure).
        """
        return (
            a >= 1
        )  # GtE not mapped directly in test? wait, in python AST it is ast.GtE, but in parser it's not checked!

    with pytest.raises(ValueError):
        parser.parse(invalid_cmp)

    def invalid_cmp_multi(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test multi-comparison/chained comparison statements, which are unsupported.

        Args:
            a (Var): The input variable.

        Returns:
            bool: Multi-comparison result (will trigger parser failure).
        """
        return 1 < a < 3

    with pytest.raises(ValueError):
        parser.parse(invalid_cmp_multi)

    def list_comp(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test list comprehensions inside script parsing, which is unsupported.

        Args:
            a (Iterable): An iterable sequence.

        Returns:
            list: A newly constructed list (will trigger parser failure).
        """
        return [x + 1 for x in a]

    with pytest.raises(ValueError):
        parser.parse(list_comp)

    def if_exp(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test parser handling of Python's conditional expressions (inline if-else).

        Args:
            a (Var): The input variable.

        Returns:
            Var: The conditional value depending on input.
        """
        return a if a > 0 else a + 1

    parser.parse(if_exp)

    def loop_for(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test parser handling of Python 'for' loops iterating over a tensor.

        Args:
            a (Var): The input iterable/tensor.

        Returns:
            Var: Accumulated sum over the iteration.
        """
        # We need a tensor to iterate
        res = a
        for i in a:
            res = res + i
        return res

    parser.parse(loop_for)

    def loop_while(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test parser handling of Python 'while' loops with dynamic conditions.

        Args:
            a (Var): The input variable.

        Returns:
            Var: Accumulated sum after loop termination.
        """
        res = a
        while res < 10:
            res = res + 1
        return res

    parser.parse(loop_while)

    def invalid_func() -> None:
        """An empty or invalid function definition to test error handling when parsed.

        Returns:
            None
        """
        pass

    with pytest.raises(ValueError):
        # not a function definition if we pass lambda
        parser.parse(lambda x: x)

    # global func call
    def sub_func() -> int:
        """A dummy ONNX-registered sub-function for testing external calls.

        Returns:
            int: A constant value of 1.
        """
        return 1

    sub_func._is_onnx_script = True
    sub_func.to_builder = lambda: GraphBuilder("sub")

    def call_func(
        a: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test nested or external function calls from within a parsed script.

        Args:
            a (Var): The input variable.

        Returns:
            Var: Result of the sub-function call.
        """
        return sub_func()

    parser.globals_dict["sub_func"] = sub_func
    parser.parse(call_func)
