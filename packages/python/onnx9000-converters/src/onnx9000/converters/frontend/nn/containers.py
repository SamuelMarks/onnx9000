"""Container layers."""

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Protocol, Union

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter
from onnx9000.core.exceptions import ConversionError


class SliceProtocol(Protocol):
    """Protocol for slice objects indexing ModuleList."""

    start: int | None
    stop: int | None
    step: int | None


class Sequential(Module):
    """A sequential container."""

    def __init__(
        self, *args: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Initialize Sequential with configured parameters and state.

        Args:
            *args: Variadic positional arguments for args.
        """
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(
        self, input: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on Sequential.

        Args:
            input: Parameter specifying input (Union[int, float, str, bool, bytes, list, dict, tuple, None]).

        Returns:
            Result of forward (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
        """
        for module in self.children():
            input = module(input)
        return input


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        """Initialize ModuleList with configured parameters and state.

        Args:
            modules: Parameter specifying modules (Iterable[Module] | None).
        """
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        """Execute append on ModuleList.

        Args:
            module: Parameter specifying module (Module).

        Returns:
            Result of append ('ModuleList').
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """Execute extend on ModuleList.

        Args:
            modules: Parameter specifying modules (Iterable[Module]).

        Returns:
            Result of extend ('ModuleList').
        """
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def __iter__(self) -> Iterator[Module]:
        """Return an iterator over elements in ModuleList.

        Returns:
            Result of iter (Iterator[Module]).
        """
        return iter(self.children())

    def __len__(self) -> int:
        """Return the number of elements contained in ModuleList.

        Returns:
            Result of len (int).
        """
        return len(self._modules)

    def __getitem__(self, idx: int | SliceProtocol) -> Union[Module, "ModuleList"]:
        """Retrieve the item or slice associated with the specified key from ModuleList.

        Args:
            idx: Parameter specifying idx (int | slice).

        Returns:
            Result of getitem (Union[Module, 'ModuleList']).
        """
        if isinstance(idx, slice):
            return ModuleList(list(self.children())[idx])
        else:
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]


class ModuleDict(Module):
    """Holds submodules in a dictionary."""

    def __init__(
        self, modules: int | float | str | bool | bytes | list | dict | tuple | None | None = None
    ) -> None:
        """Initialize ModuleDict with configured parameters and state.

        Args:
            modules: Parameter specifying modules (Union[int, float, str, bool, bytes, list, dict, tuple, None] | None).
        """
        super().__init__()
        if modules is not None:
            self.update(modules)

    def update(
        self, modules: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> "ModuleDict":
        """Execute update on ModuleDict.

        Args:
            modules: Parameter specifying modules (Union[int, float, str, bool, bytes, list, dict, tuple, None]).

        Returns:
            Result of update ('ModuleDict').

        Raises:
            TypeError: If an invalid state or argument is encountered during execution.
        """
        if not isinstance(modules, (dict, list, tuple)):
            raise ConversionError(
                "ModuleDict.update should be called with an iterable of key/value pairs, mapping, or dict"
            )
        if isinstance(modules, (list, tuple)):
            modules = dict(modules)
        for key, module in modules.items():
            self.add_module(key, module)
        return self

    def keys(self) -> Iterable[str]:
        """Execute keys on ModuleDict.

        Returns:
            Result of keys (Iterable[str]).
        """
        return self._modules.keys()

    def items(self) -> Iterable[int | float | str | bool | bytes | list | dict | tuple | None]:
        """Execute items on ModuleDict.

        Returns:
            Result of items (Iterable[Union[int, float, str, bool, bytes, list, dict, tuple, None]]).
        """
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """Execute values on ModuleDict.

        Returns:
            Result of values (Iterable[Module]).
        """
        return self._modules.values()

    def __getitem__(self, key: str) -> Module:
        """Retrieve the item or slice associated with the specified key from ModuleDict.

        Args:
            key: Parameter specifying key (str).

        Returns:
            Result of getitem (Module).
        """
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Assign a value to the specified key or slice in ModuleDict.

        Args:
            key: Parameter specifying key (str).
            module: Parameter specifying module (Module).
        """
        self.add_module(key, module)


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, parameters: Iterable[Parameter] | None = None) -> None:
        """Initialize ParameterList with configured parameters and state.

        Args:
            parameters: Parameter specifying parameters (Iterable[Parameter] | None).
        """
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def append(self, parameter: Parameter) -> "ParameterList":
        """Execute append on ParameterList.

        Args:
            parameter: Parameter specifying parameter (Parameter).

        Returns:
            Result of append ('ParameterList').
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable[Parameter]) -> "ParameterList":
        """Execute extend on ParameterList.

        Args:
            parameters: Parameter specifying parameters (Iterable[Parameter]).

        Returns:
            Result of extend ('ParameterList').
        """
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def __iter__(self) -> Iterator[Parameter]:
        """Return an iterator over elements in ParameterList.

        Returns:
            Result of iter (Iterator[Parameter]).
        """
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """Return the number of elements contained in ParameterList.

        Returns:
            Result of len (int).
        """
        return len(self._parameters)

    def __getitem__(self, idx: int) -> Parameter:
        """Retrieve the item or slice associated with the specified key from ParameterList.

        Args:
            idx: Parameter specifying idx (int).

        Returns:
            Result of getitem (Parameter).
        """
        if idx < 0:
            idx += len(self)
        return self._parameters[str(idx)]
