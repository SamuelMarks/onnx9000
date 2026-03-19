from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union


class Type:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    """Base class for all types in the Relay type system."""

    pass


@dataclass(eq=False)
class TensorType(Type):
    """A tensor type with shape and dtype."""

    shape: tuple[Union[int, str], ...]
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
