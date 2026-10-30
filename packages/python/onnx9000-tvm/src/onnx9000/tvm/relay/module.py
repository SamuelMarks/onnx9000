"""TVM submodule for AST and optimization."""

from onnx9000.core.exceptions import CompilationError

from .expr import Function, Var


class IRModule:
    """An IRModule is a collection of functions."""

    def __init__(self, functions: dict[Var, Function] | None = None) -> None:
        """Initialize IRModule with configured parameters and state.

        Args:
            functions: Parameter specifying functions (dict[Var, Function]).
        """
        """Initialize."""
        """Do the function."""
        self.functions: dict[Var, Function] = functions or {}

    def update(self, other: "IRModule") -> None:
        """Execute update on IRModule.

        Args:
            other: Parameter specifying other ('IRModule').
        """
        self.functions.update(other.functions)

    def add(self, var: Var, func: Function, update: bool = False) -> None:
        """Execute add on IRModule.

        Args:
            var: Parameter specifying var (Var).
            func: Parameter specifying func (Function).
            update: Parameter specifying update (bool).

        Raises:
            ValueError: If an invalid state or argument is encountered during execution.
        """
        if var in self.functions and not update:
            raise CompilationError(f"Function {var.name_hint} already exists in module.")
        self.functions[var] = func
