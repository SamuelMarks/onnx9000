"""
Base Module framework mirroring PyTorch nn.Module.
"""

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Callable, Optional, Union
from onnx9000.frontend.frontend.tensor import Parameter, Tensor


class Module:
    """Base class for all neural network modules."""

    def __init__(self) -> None:
        """Initialize the Module."""
        self._parameters: dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: dict[str, Optional[Module]] = OrderedDict()
        self.training: bool = True

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Registers a parameter to the module, making it available in state_dict and parameters()."""
        if not isinstance(name, str):
            raise TypeError("parameter name should be a string.")
        if param is not None and (not isinstance(param, Parameter)):
            raise TypeError("cannot assign to parameter, must be a Parameter or None")
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Registers a buffer to the module, which is persistent state but not a parameter."""
        if not isinstance(name, str):
            raise TypeError("buffer name should be a string.")
        if tensor is not None and (not isinstance(tensor, Tensor)):
            raise TypeError("cannot assign to buffer, must be a Tensor or None")
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        """Adds a child module to the current module, making it accessible via named_modules()."""
        if not isinstance(name, str):
            raise TypeError("module name should be a string.")
        if module is not None and (not isinstance(module, Module)):
            raise TypeError(f"module should be a Module, but got {type(module)}")
        self._modules[name] = module

    def _named_members(
        self,
        get_members_fn: Callable[["Module"], dict[str, Any]],
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        """Internal utility to yield named members (parameters, buffers) of the module recursively."""
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members.items():
                if v is None:
                    continue
                name = module_prefix + ("." if module_prefix else "") + k
                yield (name, v)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Yields an iterator over module parameters, yielding both the name and the parameter itself."""

        def get_parameters(m: Module) -> dict[str, Any]:
            """Tests the corresponding module functionality."""
            return m._parameters

        return self._named_members(get_parameters, prefix, recurse)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters, useful for passing to an optimizer."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Tensor]]:
        """Yields an iterator over module buffers, yielding both the name and the buffer itself."""

        def get_buffers(m: Module) -> dict[str, Any]:
            """Tests the corresponding module functionality."""
            return m._buffers

        return self._named_members(get_buffers, prefix, recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns an iterator over module buffers."""
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def children(self) -> Iterator["Module"]:
        """Returns an iterator over immediate children modules."""
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        """Yields an iterator over immediate children modules, yielding both the name and the module itself."""
        for name, module in self._modules.items():
            if module is not None:
                yield (name, module)

    def modules(self) -> Iterator["Module"]:
        """Returns an iterator over all modules in the network, including the current module."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self, memo: Optional[set["Module"]] = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[tuple[str, "Module"]]:
        """Yields an iterator over all modules in the network, yielding both the name and the module itself."""
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield (prefix, self)
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def state_dict(
        self,
        destination: Optional[dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Returns a dictionary containing the entire state of the module (parameters and buffers)."""
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Copies parameters and buffers from state_dict into this module and its descendants."""
        local_state = dict(self.state_dict(keep_vars=True).items())
        for key, param in state_dict.items():
            if key in local_state:
                local_param = local_state[key]
                if isinstance(local_param, Tensor):
                    local_param.data = param
            elif strict:
                raise RuntimeError(f"Unexpected key '{key}' in state_dict")
        if strict:
            for key in local_state:
                if key not in state_dict:
                    raise RuntimeError(f"Missing key '{key}' in state_dict")

    def to(self, *args: Any, **kwargs: Any) -> "Module":
        """Moves and/or casts the parameters and buffers of the module to the specified device or dtype."""
        for t in self.parameters():
            if t is not None:
                t.to(*args, **kwargs)
        for t in self.buffers():
            if t is not None:
                t.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "Module":
        """Sets the module in training mode, affecting layers like Dropout and BatchNorm."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Sets the module in evaluation mode."""
        return self.train(False)

    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        """Applies a given function recursively to every submodule as well as the current module."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def zero_grad(self) -> None:
        """Sets the gradients of all model parameters to None."""
        for p in self.parameters():
            if p is not None:
                p.grad = None

    def __setattr__(self, name: str, value: Union["Module", Parameter, Tensor, Any]) -> None:
        """Overrides default attribute setting to properly handle parameters, buffers, and submodules."""
        if name in ("_parameters", "_buffers", "_modules", "training"):
            object.__setattr__(self, name, value)
            return
        if not hasattr(self, "_parameters"):
            raise AttributeError("cannot assign before Module.__init__() call")
        if isinstance(value, Parameter):
            if name in self.__dict__:
                del self.__dict__[name]
            if name in self._buffers:
                del self._buffers[name]
            if name in self._modules:
                del self._modules[name]
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if name in self.__dict__:
                del self.__dict__[name]
            if name in self._parameters:
                del self._parameters[name]
            if name in self._buffers:
                del self._buffers[name]
            self.add_module(name, value)
        elif isinstance(value, Tensor):
            if name in self.__dict__:
                del self.__dict__[name]
            if name in self._parameters:
                del self._parameters[name]
            if name in self._modules:
                del self._modules[name]
            self.register_buffer(name, value)
        else:
            if name in self._parameters:
                del self._parameters[name]
            if name in self._buffers:
                del self._buffers[name]
            if name in self._modules:
                del self._modules[name]
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Overrides default attribute access to retrieve parameters, buffers, and submodules."""
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_buffers" in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Defines the computation performed at every call. Should be overridden by all subclasses."""
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invokes the forward pass of the module with the provided arguments."""
        return self.forward(*args, **kwargs)
