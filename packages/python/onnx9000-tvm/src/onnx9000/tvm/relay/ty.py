"""TVM submodule for AST and optimization."""

from dataclasses import dataclass


class Type:
    """Core class for TVM AST node or pass."""

    def __hash__(self) -> int:
        """Execute hash on Type.

        Returns:
            Result of hash.
        """
        """Do the function."""
        return id(self)

    def __eq__(self, other: int | float | str | bool | bytes | list | dict | tuple | None) -> bool:
        """Execute eq on Type.

        Args:
            other: Parameter specifying other.

        Returns:
            Result of eq.
        """
        """Do the function."""
        return self is other

    """Base class for all types in the Relay type system."""

    __dummy__ = True


@dataclass(eq=False)
class TensorType(Type):
    """A tensor type with shape and dtype."""

    shape: tuple[int | str, ...]
    dtype: str


@dataclass(eq=False)
class TupleType(Type):
    """A tuple type."""

    fields: list[Type]


@dataclass(eq=False)
class FuncType(Type):
    """A function type."""

    arg_types: list[Type]
    ret_type: Type
