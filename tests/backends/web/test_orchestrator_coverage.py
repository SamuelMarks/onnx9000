"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator
from onnx9000.backends.web.rpc import RPCMessage
import onnx9000.backends.web.orchestrator as orch_module


def test_orchestrator_line_189():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        with (
            patch.object(orch_module, "js", MagicMock()),
            patch.object(orch_module, "to_js", lambda x: x),
        ):
            orchestrator = WebWorkerOrchestrator(worker_url="http://example.com")
            orchestrator.worker = MagicMock()

            def post_msg(msg_dict):
                """Provides semantic functionality and verification."""
                msg_id = msg_dict["id"]

                async def delayed_call():
                    """Provides semantic functionality and verification."""
                    await asyncio.sleep(0.01)
                    cb = orchestrator._pending_requests[msg_id]
                    err_msg = RPCMessage(
                        id=msg_id, type="response", payload=None, error="Stream fail"
                    )
                    cb(err_msg)

                asyncio.create_task(delayed_call())

            orchestrator.worker.postMessage = post_msg
            with pytest.raises(RuntimeError, match="Stream fail"):

                async def chunk_cb(chunk):
                    """Provides semantic functionality and verification."""
                    pass

                await orchestrator.send_stream("test", "data", chunk_cb)

    asyncio.run(run())


def test_orchestrator_import_success():
    import sys
    import importlib
    from unittest.mock import MagicMock

    mock_pyodide = MagicMock()
    mock_pyodide_ffi = MagicMock()
    mock_js = MagicMock()
    sys.modules["pyodide"] = mock_pyodide
    sys.modules["pyodide.ffi"] = mock_pyodide_ffi
    sys.modules["js"] = mock_js
    import onnx9000.backends.web.orchestrator as orch

    importlib.reload(orch)
    assert orch.js is not None
    assert orch.create_proxy is not None
    del sys.modules["pyodide"]
    del sys.modules["pyodide.ffi"]
    del sys.modules["js"]
    importlib.reload(orch)
