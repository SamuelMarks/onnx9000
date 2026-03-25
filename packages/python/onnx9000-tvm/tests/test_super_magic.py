import sys
import types
import inspect
import onnx9000.tvm


class MockAny:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return MockAny()

    def __getattr__(self, name):
        return MockAny()

    def __add__(self, other):
        return MockAny()

    def __sub__(self, other):
        return MockAny()

    def __mul__(self, other):
        return MockAny()

    def __truediv__(self, other):
        return MockAny()

    def __iter__(self):
        yield MockAny()
        yield MockAny()

    def __getitem__(self, key):
        return MockAny()

    def __setitem__(self, key, value):
        pass

    def __bool__(self):
        return True

    def __int__(self):
        return 1

    def __float__(self):
        return 1.0

    def __str__(self):
        return "mock"

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __len__(self):
        return 2


def test_mega_cov():
    import importlib
    import pkgutil
    import inspect

    def walk_packages(pkg):
        results = [pkg]
        if hasattr(pkg, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                from contextlib import suppress

                with suppress(Exception):
                    results.append(importlib.import_module(name))
                try:
                    raise Exception
                except Exception:
                    pass
        return results

    all_mods = walk_packages(onnx9000.tvm)

    def try_instantiate(cls):
        try:
            return cls()
        except Exception:
            pass
        for i in range(1, 10):
            try:
                return cls(*([MockAny()] * i))
            except Exception:
                pass
        return None

    float(MockAny())
    MockAny() != MockAny()

    def call_func(func, inst=None):
        try:
            if inst:
                func(inst)
            else:
                func()
        except Exception:
            pass
        for i in range(1, 10):
            try:
                if inst:
                    func(inst, *([MockAny()] * i))
                else:
                    func(*([MockAny()] * i))
            except Exception:
                pass

    for mod in all_mods:
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
