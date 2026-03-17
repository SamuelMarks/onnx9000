"""Container layers."""

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any, Optional, Union
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter


class Sequential(Module):
    """A sequential container."""

    def __init__(self, *args: Any) -> None:
        """Implements the __init__ method."""
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input: Any) -> Any:
        """Implements the forward method."""
        for module in self.children():
            input = module(input)
        return input


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        """Implements the __init__ method."""
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        """Implements the append method."""
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """Implements the extend method."""
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def __iter__(self) -> Iterator[Module]:
        """Implements the __iter__ method."""
        return iter(self.children())

    def __len__(self) -> int:
        """Implements the __len__ method."""
        return len(self._modules)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
        """Implements the __getitem__ method."""
        if isinstance(idx, slice):
            return ModuleList(list(self.children())[idx])
        else:
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]


class ModuleDict(Module):
    """Holds submodules in a dictionary."""

    def __init__(self, modules: Optional[Any] = None) -> None:
        """Implements the __init__ method."""
        super().__init__()
        if modules is not None:
            self.update(modules)

    def update(self, modules: Any) -> "ModuleDict":
        """Implements the update method."""
        if not isinstance(modules, (dict, list, tuple)):
            raise TypeError(
                "ModuleDict.update should be called with an iterable of key/value pairs, mapping, or dict"
            )
        if isinstance(modules, (list, tuple)):
            modules = dict(modules)
        for key, module in modules.items():
            self.add_module(key, module)
        return self

    def keys(self) -> Iterable[str]:
        """Implements the keys method."""
        return self._modules.keys()

    def items(self) -> Iterable[Any]:
        """Implements the items method."""
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """Implements the values method."""
        return self._modules.values()

    def __getitem__(self, key: str) -> Module:
        """Implements the __getitem__ method."""
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Implements the __setitem__ method."""
        self.add_module(key, module)


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, parameters: Optional[Iterable[Parameter]] = None) -> None:
        """Implements the __init__ method."""
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def append(self, parameter: Parameter) -> "ParameterList":
        """Implements the append method."""
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable[Parameter]) -> "ParameterList":
        """Implements the extend method."""
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def __iter__(self) -> Iterator[Parameter]:
        """Implements the __iter__ method."""
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """Implements the __len__ method."""
        return len(self._parameters)

    def __getitem__(self, idx: int) -> Parameter:
        """Implements the __getitem__ method."""
        if idx < 0:
            idx += len(self)
        return self._parameters[str(idx)]
