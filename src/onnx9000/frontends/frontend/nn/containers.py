"""Container layers."""

from typing import Any, Iterable, Union, Iterator, Optional
from collections import OrderedDict
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter


class Sequential(Module):
    """A sequential container."""

    def __init__(self, *args: Any) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input: Any) -> Any:
        """Provides semantic functionality and verification."""
        for module in self.children():
            input = module(input)
        return input


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        """Provides semantic functionality and verification."""
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """Provides semantic functionality and verification."""
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def __iter__(self) -> Iterator[Module]:
        """Provides semantic functionality and verification."""
        return iter(self.children())

    def __len__(self) -> int:
        """Provides semantic functionality and verification."""
        return len(self._modules)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
        """Provides semantic functionality and verification."""
        if isinstance(idx, slice):
            return ModuleList(list(self.children())[idx])
        else:
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]


class ModuleDict(Module):
    """Holds submodules in a dictionary."""

    def __init__(self, modules: Optional[Any] = None) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        if modules is not None:
            self.update(modules)

    def update(self, modules: Any) -> "ModuleDict":
        """Provides semantic functionality and verification."""
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
        """Provides semantic functionality and verification."""
        return self._modules.keys()

    def items(self) -> Iterable[Any]:
        """Provides semantic functionality and verification."""
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """Provides semantic functionality and verification."""
        return self._modules.values()

    def __getitem__(self, key: str) -> Module:
        """Provides semantic functionality and verification."""
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Provides semantic functionality and verification."""
        self.add_module(key, module)


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, parameters: Optional[Iterable[Parameter]] = None) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def append(self, parameter: Parameter) -> "ParameterList":
        """Provides semantic functionality and verification."""
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable[Parameter]) -> "ParameterList":
        """Provides semantic functionality and verification."""
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def __iter__(self) -> Iterator[Parameter]:
        """Provides semantic functionality and verification."""
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """Provides semantic functionality and verification."""
        return len(self._parameters)

    def __getitem__(self, idx: int) -> Parameter:
        """Provides semantic functionality and verification."""
        if idx < 0:
            idx += len(self)
        return self._parameters[str(idx)]
