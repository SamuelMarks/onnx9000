import pytest


def test_rpc_import_success():
    import sys
    import importlib
    from unittest.mock import MagicMock

    mock_pyodide = MagicMock()
    mock_pyodide_ffi = MagicMock()
    mock_js = MagicMock()
    sys.modules["pyodide"] = mock_pyodide
    sys.modules["pyodide.ffi"] = mock_pyodide_ffi
    sys.modules["js"] = mock_js
    import onnx9000.backends.web.rpc as rpc

    importlib.reload(rpc)
    assert rpc.js is not None
    assert rpc.to_js is not None
    del sys.modules["pyodide"]
    del sys.modules["pyodide.ffi"]
    del sys.modules["js"]
    importlib.reload(rpc)
