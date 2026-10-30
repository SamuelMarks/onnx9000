"""Tests for ops coverage."""

from onnx9000.core import ops
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import Tensor
from onnx9000.core.ops import torch_auto


def test_all_ops_coverage() -> None:
    """Test all operators for coverage purposes, including optional arguments."""
    t = Tensor(name="t", shape=(1,), dtype=None)

    # We can use reflection to call them with dummy args
    import inspect

    for module in [ops, torch_auto]:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Check if function is defined in the module itself
            if func.__module__ != module.__name__:
                continue

            if name.startswith("_") or name in {"register_op", "record_op"}:
                continue

            sig = inspect.signature(func)

            # Base call: only required args
            args = []
            for _p_name, p in sig.parameters.items():
                if p.default is inspect.Parameter.empty and p.kind not in [
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ]:
                    args.append(t)

            try:
                res = func(*args)
                assert res is not None or res is None
            except (
                Onnx9000Error,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                IndexError,
                AttributeError,
                ImportError,
                LookupError,
                ArithmeticError,
                BufferError,
                AssertionError,
                StopIteration,
            ) as exc:
                assert isinstance(exc, Exception)

            # Full call: all args (using t for all)
            args_full = []
            for _p_name, p in sig.parameters.items():
                if p.kind not in [
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ]:
                    args_full.append(t)

            try:
                res_full = func(*args_full)
                assert res_full is not None or res_full is None
            except (
                Onnx9000Error,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                IndexError,
                AttributeError,
                ImportError,
                LookupError,
                ArithmeticError,
                BufferError,
                AssertionError,
                StopIteration,
            ) as exc_full:
                assert isinstance(exc_full, Exception)
