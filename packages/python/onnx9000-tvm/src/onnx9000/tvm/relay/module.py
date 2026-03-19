from typing import Dict

from .expr import Function, Var


class IRModule:
    """An IRModule is a collection of functions."""

    def __init__(self, functions: dict[Var, Function] = None):
        self.functions: dict[Var, Function] = functions or {}

    def update(self, other: "IRModule"):
        self.functions.update(other.functions)

    def add(self, var: Var, func: Function, update: bool = False):
        if var in self.functions and not update:
            raise ValueError(f"Function {var.name_hint} already exists in module.")
        self.functions[var] = func
