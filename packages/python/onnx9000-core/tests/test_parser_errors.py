import pytest
from pathlib import Path
import tempfile
from onnx9000.core.parser.core import load, from_bytes
from onnx9000.core.exceptions import ONNXParseError
from onnx9000.core import onnx_pb2


def test_load_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.onnx"

        # Too small
        with open(path, "wb") as f:
            f.write(b"123")
        with pytest.raises(ONNXParseError, match="too small"):
            load(path)

        # Invalid protobuf
        with open(path, "wb") as f:
            f.write(b"1234567890")
        with pytest.raises(ONNXParseError, match="Failed to parse"):
            load(path)

        # ir_version == 0
        mp = onnx_pb2.ModelProto()
        mp.ir_version = 0
        mp.producer_name = "this is a very long string to bypass length check"
        with open(path, "wb") as f:
            f.write(mp.SerializeToString())
        with pytest.raises(ONNXParseError, match="ir_version is 0"):
            load(path)


def test_from_bytes_errors():
    # Too small
    with pytest.raises(ONNXParseError, match="too small"):
        from_bytes(b"123")

    # Invalid protobuf
    with pytest.raises(ONNXParseError, match="Failed to parse"):
        from_bytes(b"1234567890")

    # ir_version == 0
    mp = onnx_pb2.ModelProto()
    mp.ir_version = 0
    mp.producer_name = "this is a very long string to bypass length check"
    with pytest.raises(ONNXParseError, match="ir_version is 0"):
        from_bytes(mp.SerializeToString())
