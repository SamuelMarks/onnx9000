import onnx9000.core.registry as registry
import torch
import torch.fx
from onnx9000.converters.torch.fx import FXParser
from onnx9000.converters.torch.script import TorchScriptParser


def test_fx_dynamic_shape():
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    # inject dynamic dim
    for node in traced.graph.nodes:
        node.meta["tensor_meta"] = type(
            "TensorMeta", (), {"shape": (1, "batch", 3), "dtype": torch.float32}
        )()
    p = FXParser(traced)
    p.parse()


def test_fx_call_method():
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    p = FXParser(traced)
    p.parse()


def test_fx_get_attr():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("my_buf", torch.zeros(1))
            self.param = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x + self.my_buf + self.param

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    p = FXParser(traced)
    p.parse()


def test_fx_call_module():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    p = FXParser(traced)
    p.parse()


def test_fx_fallback():
    class Mod(torch.nn.Module):
        def forward(self, x, y):
            # uses mul and relu to trigger fallback
            return torch.relu(torch.mul(x, y))

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    old_get = registry.global_registry.get_op

    def fake_get(*args, **kwargs):
        raise Exception("Fake")

    registry.global_registry.get_op = fake_get
    try:
        p = FXParser(traced)
        p.parse()
    finally:
        registry.global_registry.get_op = old_get


def test_fx_multiple_outputs():
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x, x + 1

    m = Mod()
    traced = torch.fx.symbolic_trace(m)
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            node.meta["tensor_meta"] = type(
                "TensorMeta", (), {"shape": None, "dtype": torch.float32}
            )()
    p = FXParser(traced)
    p.parse()

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
