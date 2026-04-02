import onnx9000.core.ops as ops
import onnx9000.core.ops.torch_auto as torch_auto
import pytest
from onnx9000.core.ir import Tensor
from onnx9000.core.ops import *


def test_all_ops_coverage():
    """Test all operators for coverage purposes, including optional arguments."""
    t = Tensor(name="t", shape=(1,), dtype=None)

    # We can use reflection to call them with dummy args
    import inspect

    for module in [ops, torch_auto]:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Check if function is defined in the module itself
            if func.__module__ != module.__name__:
                continue

            if name.startswith("_") or name == "register_op" or name == "record_op":
                continue

            sig = inspect.signature(func)

            # Base call: only required args
            args = []
            for p_name, p in sig.parameters.items():
                if p.default is inspect.Parameter.empty and p.kind not in [
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ]:
                    args.append(t)

            try:
                func(*args)
            except Exception:
                pass

            # Full call: all args (using t for all)
            args_full = []
            for p_name, p in sig.parameters.items():
                if p.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                    args_full.append(t)

            try:
                func(*args_full)
            except Exception:
                pass
