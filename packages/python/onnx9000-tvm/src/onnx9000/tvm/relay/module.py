"""TVM submodule for AST and optimization."""

from .expr import Function, Var


class IRModule:
    """An IRModule is a collection of functions."""

    def __init__(self, functions: dict[Var, Function] = None):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.functions: dict[Var, Function] = functions or {}

    def update(self, other: "IRModule"):
        """Do the function."""
        self.functions.update(other.functions)

    def add(self, var: Var, func: Function, update: bool = False):
        """Do the function."""
        if var in self.functions and not update:
            raise ValueError(f"Function {var.name_hint} already exists in module.")
        self.functions[var] = func
