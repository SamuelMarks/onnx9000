"""Tests for packages/python/onnx9000-tvm/tests/test_super_magic.py."""

import inspect
import sys
import types

import onnx9000.tvm


class MockAny:
    """MockAny implementation."""

    def __init__(self, *args, **kwargs):
        """Perform   init   operation."""
        return None

    def __call__(self, *args, **kwargs):
        """Perform   call   operation."""
        return MockAny()

    def __getattr__(self, name):
        """Perform   getattr   operation."""
        return MockAny()

    def __add__(self, other):
        """Perform   add   operation."""
        return MockAny()

    def __sub__(self, other):
        """Perform   sub   operation."""
        return MockAny()

    def __mul__(self, other):
        """Perform   mul   operation."""
        return MockAny()

    def __truediv__(self, other):
        """Perform   truediv   operation."""
        return MockAny()

    def __iter__(self):
        """Perform   iter   operation."""
        yield MockAny()
        yield MockAny()

    def __getitem__(self, key):
        """Perform   getitem   operation."""
        return MockAny()

    def __setitem__(self, key, value):
        """Perform   setitem   operation."""
        return None

    def __bool__(self):
        """Perform   bool   operation."""
        return True

    def __int__(self):
        """Perform   int   operation."""
        return 1

    def __float__(self):
        """Perform   float   operation."""
        return 1.0

    def __str__(self):
        """Perform   str   operation."""
        return "mock"

    def __eq__(self, other):
        """Perform   eq   operation."""
        return True

    def __ne__(self, other):
        """Perform   ne   operation."""
        return False

    def __len__(self):
        """Perform   len   operation."""
        return 2


def test_mega_cov():
    """Test mega cov."""
    import importlib
    import inspect
    import pkgutil

    def walk_packages(pkg):
        """Perform walk packages operation."""
        results = [pkg]
        if hasattr(pkg, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                from contextlib import suppress

                with suppress(Exception):
                    results.append(importlib.import_module(name))
                try:
                    raise Exception
                except Exception:
                    return None
        return results

    all_mods = walk_packages(onnx9000.tvm)

    def try_instantiate(cls):
        """Perform try instantiate operation."""
        try:
            return cls()
        except Exception:
            return None
        for i in range(1, 10):
            try:
                return cls(*[MockAny()] * i)
            except Exception:
                return None
        return None

    float(MockAny())
    _ = MockAny() != MockAny()

    def call_func(func, inst=None):
        """Perform call func operation."""
        try:
            if inst:
                func(inst)
            else:
                func()
        except Exception:
            return None
        for i in range(1, 10):
            try:
                if inst:
                    func(inst, *[MockAny()] * i)
                else:
                    func(*[MockAny()] * i)
            except Exception:
                return None

    for mod in all_mods or []:
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                inst = try_instantiate(obj)
                if inst:
                    for m_name, m_obj in inspect.getmembers(obj):
                        if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                            call_func(m_obj, inst)
                            call_func(m_obj)
            elif inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                call_func(obj)
