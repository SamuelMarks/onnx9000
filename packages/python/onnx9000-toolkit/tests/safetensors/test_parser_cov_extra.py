"""Tests for packages/python/onnx9000-toolkit/tests/safetensors/test_parser_cov_extra.py."""

import pytest
import os
import sys
import ctypes
import json
import struct
from unittest.mock import patch, MagicMock
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
    SafetensorsInvalidJSONError,
    load,
    save_sharded,
)
from onnx9000.toolkit.safetensors.parser import save_file


def test_mmap_madvise_exception(tmp_path):
    """Test mmap madvise exception."""
    path = tmp_path / "model.safetensors"
    save_file({"a": b"1234"}, str(path))
    import mmap

    original_mmap = mmap.mmap

    class MockMmap(original_mmap):
        """MockMmap implementation."""

        def madvise(self, *args, **kwargs):
            """Perform madvise operation."""
            raise Exception("mocked error")

    with patch("mmap.mmap", new=MockMmap):
        st = SafeTensors(str(path), mmap_hint=True)
        st._close()


def test_json_recursion_error():
    """Test json recursion error."""
    with patch("json.loads", side_effect=RecursionError("too deep")):
        with pytest.raises(
            SafetensorsInvalidJSONError, match="JSON deeply nested recursion limits reached"
        ):
            header = b'{"__metadata__": {}}'
            buf = struct.pack("<Q", len(header)) + header
            SafeTensors(buf)


def test_metadata_missing_format(tmp_path, caplog):
    """Test metadata missing format."""
    import logging

    path = tmp_path / "model.safetensors"
    metadata = {"other": "value"}
    header_dict = {
        "__metadata__": metadata,
        "a": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
    }
    header_bytes = json.dumps(header_dict).encode("utf-8")
    tensor_bytes = b"1234"
    with open(str(path), "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(tensor_bytes)
    with caplog.at_level(logging.WARNING):
        st = SafeTensors(str(path))
        st._close()
    assert "Safetensors metadata is missing standard HuggingFace 'format' key." in caplog.text


def test_virtuallock_failure(tmp_path, caplog):
    """Test virtuallock failure."""
    import logging

    path = tmp_path / "model.safetensors"
    save_file({"a": b"1234"}, str(path))
    with patch("sys.platform", "win32"):
        mock_kernel32 = MagicMock()
        mock_kernel32.VirtualLock.return_value = 0
        with patch("ctypes.windll", MagicMock(kernel32=mock_kernel32), create=True):
            with caplog.at_level(logging.WARNING):
                st = SafeTensors(str(path))
                st.get_pinned_tensor("a")
                st._close()
            assert "VirtualLock failed for tensor a" in caplog.text


def test_load_regex(tmp_path):
    """Test load regex."""
    path = tmp_path / "model.safetensors"
    save_file({"a": b"1234", "b": b"5678"}, str(path))
    with open(str(path), "rb") as f:
        data = f.read()
    res = load(data, pattern="^a$")
    assert "a" in res
    assert "b" not in res
    with patch.object(SafeTensors, "keys", return_value=[123]):
        with pytest.raises(TypeError, match="Dictionary keys must be strings"):
            load(data)


def test_save_sharded_raw_data(tmp_path):
    """Test save sharded raw data."""

    class DummyData:
        """DummyData implementation."""

        def __init__(self, size):
            """Perform   init   operation."""
            self.raw_data = b"x" * size
            self.dims = [size]

        def __len__(self):
            """Perform   len   operation."""
            return len(self.raw_data)

    dummy = DummyData(5)
    len(dummy)
    tensors = {"a": dummy}
    save_sharded(tensors, str(tmp_path / "test"), max_shard_size=10)
    assert os.path.exists(str(tmp_path / "test" / "model-00001-of-00001.safetensors"))


def test_convert_pytorch_to_safetensors_mocked(tmp_path):
    """Test convert pytorch to safetensors mocked."""
    import sys
    from unittest.mock import MagicMock, patch

    mock_torch = MagicMock()

    class DummyTensor:
        """DummyTensor implementation."""

        def numpy(self):
            """Perform numpy operation."""
            import numpy as np

            return np.array([1.0])

    mock_torch.Tensor = DummyTensor
    mock_torch.tensor = lambda x: DummyTensor()
    state_dict = {"a": mock_torch.tensor([1.0]), "b": 42}
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("torch.load", return_value=state_dict) as mock_load:
            from onnx9000.toolkit.safetensors.converters import convert_pytorch_to_safetensors
            import os

            output_file = tmp_path / "model.safetensors"
            (tmp_path / "dummy.bin").write_text("")
            convert_pytorch_to_safetensors(str(tmp_path), str(output_file))
            assert os.path.exists(str(output_file))
            print("state_dict:", state_dict)
        assert mock_load.called
        print("state_dict:", state_dict)


def test_convert_tf_to_safetensors(tmp_path):
    """Test convert tf to safetensors."""
    import sys
    from unittest.mock import MagicMock, patch

    mock_tf = MagicMock()

    class DummyVar:
        """DummyVar implementation."""

        def __init__(self, name):
            """Perform   init   operation."""
            self.name = name
            import numpy as np

            self._np = np.array([2.0])

        def numpy(self):
            """Perform numpy operation."""
            return self._np

    class DummyModel:
        """DummyModel implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.variables = [DummyVar("layer1:0"), DummyVar("layer2")]

    mock_tf.saved_model.load.return_value = DummyModel()
    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        from onnx9000.toolkit.safetensors.converters import convert_tf_to_safetensors

        output_file = tmp_path / "tf_model.safetensors"
        convert_tf_to_safetensors("dummy_dir", str(output_file))
        import os

        assert os.path.exists(str(output_file))


def test_convert_pytorch_missing_torch():
    """Test convert pytorch missing torch."""
    import sys
    from onnx9000.toolkit.safetensors.converters import convert_pytorch_to_safetensors
    import pytest

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="PyTorch is required"):
            convert_pytorch_to_safetensors("dummy_dir")


def test_convert_tf_missing_tf():
    """Test convert tf missing tf."""
    import sys
    from onnx9000.toolkit.safetensors.converters import convert_tf_to_safetensors
    import pytest

    with patch.dict(sys.modules, {"tensorflow": None}):
        with pytest.raises(ImportError, match="TensorFlow is required"):
            convert_tf_to_safetensors("dummy_dir", "out")


def test_convert_pytorch_no_bins(tmp_path):
    """Test convert pytorch no bins."""
    import sys
    from unittest.mock import MagicMock, patch
    from onnx9000.toolkit.safetensors.converters import convert_pytorch_to_safetensors

    with patch.dict(sys.modules, {"torch": MagicMock()}):
        convert_pytorch_to_safetensors(str(tmp_path))
