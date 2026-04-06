import torch
import torch.fx
from onnx9000.converters.torch.fx import FXParser
from onnx9000.converters.torch.script import TorchScriptParser


def test_fx_unknown_op():
    class MyMod(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.foo_bar(x)

    # Mocking torch.ops.aten.foo_bar or just using an unsupported op
    class Mod(torch.nn.Module):
        def forward(self, x):
            return torch.add(x, 1)  # If it fails the registry

    m = Mod()
    # It will fallback. To force the exception, we can just patch global_registry
    import onnx9000.core.registry as registry

    old_get = registry.global_registry.get_op

    def fake_get(*args, **kwargs):
        raise Exception("Fake")

    registry.global_registry.get_op = fake_get
    try:
        traced = torch.fx.symbolic_trace(m)
        p = FXParser(traced)
        p.parse()
    finally:
        registry.global_registry.get_op = old_get


def test_script_unknown_op():
    import onnx9000.core.registry as registry

    old_get = registry.global_registry.get_op

    def fake_get(*args, **kwargs):
        raise Exception("Fake")

    class Mod(torch.nn.Module):
        def forward(self, x):
            return x + 1

    sm = torch.jit.script(Mod())

    registry.global_registry.get_op = fake_get
    try:
        p = TorchScriptParser(sm)
        p.parse()
    finally:
        registry.global_registry.get_op = old_get
