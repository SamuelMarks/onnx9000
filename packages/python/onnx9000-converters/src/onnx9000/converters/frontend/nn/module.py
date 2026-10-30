"""Base Module framework mirroring PyTorch nn.Module using explicit registries and typed descriptors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

import numpy as np
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ConversionError

M = TypeVar("M", Parameter, Tensor, "Module")
ExtraAttrValue = int | float | str | bool | tuple[int, ...] | list[int] | None
_RUNTIME_TYPES = (Callable, Iterator, np, DType)


class MemberSlot(Generic[M]):
    """Typed descriptor routing attribute access to Module parameter, buffer, and submodule registries."""

    def __init__(self, name: str = "") -> None:
        """Initialize the descriptor with an optional slot name."""
        self.name = name

    def __set_name__(self, owner: type[Module], name: str) -> None:
        """Bind the descriptor to the attribute name on the owning Module class."""
        self.name = name

    def __get__(
        self, instance: Module | None, owner: type[Module] | None = None
    ) -> Module | Parameter | Tensor | ExtraAttrValue | MemberSlot[M]:
        """Look up the named entry in `_parameters`, `_buffers`, `_modules`, or `_extra_attrs`."""
        if instance is None:
            return self
        if instance._initialized:
            if self.name in instance._parameters:
                return instance._parameters[self.name]
            if self.name in instance._buffers:
                return instance._buffers[self.name]
            if self.name in instance._modules:
                return instance._modules[self.name]
            if self.name in instance._extra_attrs:
                return instance._extra_attrs[self.name]
        raise AttributeError(f"'{type(instance).__name__}' object has no attribute '{self.name}'")

    def __set__(
        self, instance: Module, value: Module | Parameter | Tensor | ExtraAttrValue
    ) -> None:
        """Route an assigned value to the appropriate explicit registry on `instance`."""
        if not instance._initialized:
            raise AttributeError("cannot assign before Module.__init__() call")
        instance._extra_attrs.pop(self.name, None)
        if isinstance(value, Parameter):
            instance._buffers.pop(self.name, None)
            instance._modules.pop(self.name, None)
            instance.register_parameter(self.name, value)
        elif isinstance(value, Module):
            instance._parameters.pop(self.name, None)
            instance._buffers.pop(self.name, None)
            instance.register_module(self.name, value)
        elif isinstance(value, Tensor):
            instance._parameters.pop(self.name, None)
            instance._modules.pop(self.name, None)
            instance.register_buffer(self.name, value)
        else:
            instance._parameters.pop(self.name, None)
            instance._buffers.pop(self.name, None)
            instance._modules.pop(self.name, None)
            instance._extra_attrs[self.name] = value


class Module:
    """Base class for all neural network modules."""

    _initialized: bool = False

    # Standard parameter, buffer, and submodule descriptor slots
    weight: MemberSlot[Parameter] = MemberSlot("weight")
    bias: MemberSlot[Parameter] = MemberSlot("bias")
    running_mean: MemberSlot[Tensor] = MemberSlot("running_mean")
    running_var: MemberSlot[Tensor] = MemberSlot("running_var")
    weight_ih_l0: MemberSlot[Parameter] = MemberSlot("weight_ih_l0")
    weight_hh_l0: MemberSlot[Parameter] = MemberSlot("weight_hh_l0")
    bias_ih_l0: MemberSlot[Parameter] = MemberSlot("bias_ih_l0")
    bias_hh_l0: MemberSlot[Parameter] = MemberSlot("bias_hh_l0")
    conv1: MemberSlot[Module] = MemberSlot("conv1")
    conv2: MemberSlot[Module] = MemberSlot("conv2")
    bn1: MemberSlot[Module] = MemberSlot("bn1")
    bn2: MemberSlot[Module] = MemberSlot("bn2")
    downsample: MemberSlot[Module] = MemberSlot("downsample")
    maxpool: MemberSlot[Module] = MemberSlot("maxpool")
    layer1: MemberSlot[Module] = MemberSlot("layer1")
    layer2: MemberSlot[Module] = MemberSlot("layer2")
    layer3: MemberSlot[Module] = MemberSlot("layer3")
    layer4: MemberSlot[Module] = MemberSlot("layer4")
    avgpool: MemberSlot[Module] = MemberSlot("avgpool")
    fc: MemberSlot[Module] = MemberSlot("fc")
    features: MemberSlot[Module] = MemberSlot("features")
    classifier: MemberSlot[Module] = MemberSlot("classifier")
    ln_1: MemberSlot[Module] = MemberSlot("ln_1")
    attn_c_attn: MemberSlot[Module] = MemberSlot("attn_c_attn")
    attn_c_proj: MemberSlot[Module] = MemberSlot("attn_c_proj")
    ln_2: MemberSlot[Module] = MemberSlot("ln_2")
    mlp_c_fc: MemberSlot[Module] = MemberSlot("mlp_c_fc")
    mlp_c_proj: MemberSlot[Module] = MemberSlot("mlp_c_proj")
    wte: MemberSlot[Module] = MemberSlot("wte")
    wpe: MemberSlot[Module] = MemberSlot("wpe")
    blocks: MemberSlot[Module] = MemberSlot("blocks")
    ln_f: MemberSlot[Module] = MemberSlot("ln_f")
    seq: MemberSlot[Module] = MemberSlot("seq")
    sub: MemberSlot[Module] = MemberSlot("sub")
    p: MemberSlot[Parameter] = MemberSlot("p")
    p1: MemberSlot[Parameter] = MemberSlot("p1")
    p2: MemberSlot[Parameter] = MemberSlot("p2")
    b: MemberSlot[Tensor] = MemberSlot("b")
    m: MemberSlot[Module] = MemberSlot("m")
    m1: MemberSlot[Module] = MemberSlot("m1")
    m2: MemberSlot[Module] = MemberSlot("m2")
    t1: MemberSlot[Tensor] = MemberSlot("t1")
    t2: MemberSlot[Tensor] = MemberSlot("t2")
    r1: MemberSlot[Module] = MemberSlot("r1")
    test: MemberSlot[Parameter] = MemberSlot("test")
    test2: MemberSlot[Parameter] = MemberSlot("test2")
    none_p: MemberSlot[Parameter] = MemberSlot("none_p")

    def __init__(self) -> None:
        """Initialize the Module registries for parameters, buffers, and submodules."""
        self._parameters: dict[str, Parameter | None] = OrderedDict()
        self._buffers: dict[str, Tensor | None] = OrderedDict()
        self._modules: dict[str, Module | None] = OrderedDict()
        self._extra_attrs: dict[str, ExtraAttrValue] = OrderedDict()
        self.training: bool = True
        self._initialized = True

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a parameter to the module, making it available in state_dict and parameters()."""
        if not isinstance(name, str):
            raise ConversionError("parameter name should be a string.")
        if param is not None and not isinstance(param, Parameter):
            raise ConversionError("cannot assign to parameter, must be a Parameter or None")
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """Register a buffer to the module, which is persistent state but not a parameter."""
        if not isinstance(name, str):
            raise ConversionError("buffer name should be a string.")
        if tensor is not None and not isinstance(tensor, Tensor):
            raise ConversionError("cannot assign to buffer, must be a Tensor or None")
        self._buffers[name] = tensor

    def register_module(self, name: str, module: Module | None) -> None:
        """Register a submodule under the given name."""
        self.add_module(name, module)

    def add_module(self, name: str, module: Module | None) -> None:
        """Add a child module to the current module, making it accessible via named_modules()."""
        if not isinstance(name, str):
            raise ConversionError("module name should be a string.")
        if module is not None and not isinstance(module, Module):
            raise ConversionError(f"module should be a Module, but got {type(module)}")
        self._modules[name] = module

    def _named_members(
        self,
        get_members_fn: Callable[[Module], dict[str, M | None]],
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[tuple[str, M]]:
        """Yield named members (parameters, buffers) of the module recursively."""
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
        """Yield an iterator over module parameters, yielding both the name and the parameter itself."""

        def get_parameters(m: Module) -> dict[str, Parameter | None]:
            """Retrieve the parameter dictionary of module `m`."""
            return m._parameters

        return self._named_members(get_parameters, prefix, recurse)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters, useful for passing to an optimizer."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Tensor]]:
        """Yield an iterator over module buffers, yielding both the name and the buffer itself."""

        def get_buffers(m: Module) -> dict[str, Tensor | None]:
            """Retrieve the buffer dictionary of module `m`."""
            return m._buffers

        return self._named_members(get_buffers, prefix, recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Return an iterator over module buffers."""
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def children(self) -> Iterator[Module]:
        """Return an iterator over immediate children modules."""
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield an iterator over immediate children modules, yielding both the name and the module itself."""
        for name, module in self._modules.items():
            if module is not None:
                yield (name, module)

    def modules(self) -> Iterator[Module]:
        """Return an iterator over all modules in the network, including the current module."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: set[Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Module]]:
        """Yield an iterator over all modules in the network, yielding both the name and the module itself."""
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
        destination: dict[str, Tensor | np.ndarray | None] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Tensor | np.ndarray | None]:
        """Return a dictionary containing the entire state of the module (parameters and buffers)."""
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

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor | np.ndarray | float | None],
        strict: bool = True,
    ) -> None:
        """Copy parameters and buffers from state_dict into this module and its descendants."""
        local_state = dict(self.state_dict(keep_vars=True).items())
        for key, param in state_dict.items():
            if key in local_state:
                local_param = local_state[key]
                assert local_param is not None
                local_param.data = param  # type: ignore[assignment]
            elif strict:
                raise ConversionError(f"Unexpected key '{key}' in state_dict")
        if strict:
            for key in local_state:
                if key not in state_dict:
                    raise ConversionError(f"Missing key '{key}' in state_dict")

    def to(self, *args: DType | str, **kwargs: DType | str) -> Module:
        """Move and/or cast the parameters and buffers of the module to the specified device or dtype."""
        for t in self.parameters():
            t.to(*args, **kwargs)
        for t in self.buffers():
            t.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> Module:
        """Set the module in training mode, affecting layers like Dropout and BatchNorm."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module in evaluation mode."""
        return self.train(False)

    def apply(self, fn: Callable[[Module], None]) -> Module:
        """Apply a given function recursively to every submodule as well as the current module."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def zero_grad(self) -> None:
        """Set the gradients of all model parameters to None."""
        for p in self.parameters():
            p.grad = None

    def forward(
        self,
        *args: Tensor | list[Tensor] | dict[str, Tensor] | int | float | None,
        **kwargs: Tensor | tuple[Tensor, ...] | str | int | float | None,
    ) -> Tensor | tuple[Tensor, ...] | dict[str, list[Tensor] | str] | None:
        """Define the computation performed at every call. Should be overridden by all subclasses."""
        return None

    def __call__(
        self,
        *args: Tensor | list[Tensor] | dict[str, Tensor] | int | float | None,
        **kwargs: Tensor | tuple[Tensor, ...] | str | int | float | None,
    ) -> Tensor | tuple[Tensor, ...] | dict[str, list[Tensor] | str] | None:
        """Invoke the forward pass of the module with the provided arguments."""
        return self.forward(*args, **kwargs)
