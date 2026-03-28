"""Module providing core logic and structural definitions."""

from onnx9000.core.dtypes import DType

"Module providing core logic and structural definitions."
import os
import tempfile
from pathlib import Path

from onnx9000.converters.frontend.exporter import export
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class SimpleModel(Module):
    """Class SimpleModel implementation."""

    def forward(self, x):
        """Test the forward functionality."""
        return x * 2.0 + 1.0


def test_export_basic() -> None:
    """Tests the test_export_basic functionality."""
    model = SimpleModel()
    x = Tensor((1, 3, 224, 224), DType.FLOAT32)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.onnx"
        export(
            model,
            args=(x,),
            f=out_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
        )
        assert out_path.exists()
        assert os.path.getsize(out_path) > 0


def test_export_args_coverage(tmp_path) -> None:
    """Tests the test_export_args_coverage functionality."""
    from onnx9000.converters.frontend.exporter import export

    def my_fn(a, b):
        """Test the my_fn functionality."""
        return a + b

    t1 = Tensor((10,), DType.FLOAT32, "a")
    t2 = Tensor((10,), DType.FLOAT32, "b")
    export(
        my_fn,
        (t1, t2),
        tmp_path / "test.onnx",
        export_params=True,
        opset_version=15,
        input_names=["a", "b"],
        output_names=["c"],
        dynamic_axes={"a": {0: "batch"}, "c": {0: "batch"}},
        custom_opsets={"custom": 1},
    )


def test_export_dynamic_axes() -> None:
    """Tests the test_export_dynamic_axes functionality."""
    from onnx9000.converters.frontend.exporter import export

    def my_fn(a):
        """Test the my_fn functionality."""
        return a

    t1 = Tensor((10,), DType.FLOAT32, "a")
    export(
        my_fn,
        t1,
        "test.onnx",
        dynamic_axes={"a": {0: "batch"}},
        input_names=["a"],
        output_names=["a"],
    )


from unittest.mock import MagicMock, patch


def test_export_large_model(tmp_path) -> None:
    """Tests the test_export_large_model functionality."""
    from onnx9000.converters.frontend.exporter import export
    from onnx9000.converters.frontend.tensor import Tensor
    from onnx9000.core.dtypes import DType

    def my_fn(a):
        """Test the my_fn functionality."""
        return a

    t1 = Tensor((10,), DType.FLOAT32, "a")
    out_path = tmp_path / "large.onnx"
    with patch("onnx9000.converters.frontend.exporter.build_model_proto") as mock_build:
        mock_proto = MagicMock()
        mock_proto.ByteSize.return_value = 3 * 1024 * 1024 * 1024
        mock_proto.SerializeToString.return_value = b"large_model_data"
        mock_build.return_value = mock_proto
        export(my_fn, t1, out_path)
        with open(out_path, "rb") as f:
            assert f.read() == b"large_model_data"


def test_exporter_large_model(tmpdir) -> None:
    """Tests the test_exporter_large_model functionality."""
    from unittest.mock import patch

    import onnx9000.core.onnx_pb2 as onnx_pb2
    from onnx9000.converters.frontend.exporter import export
    from onnx9000.converters.frontend.tensor import Tensor

    def simple_func(x):
        """Test the simple_func functionality."""
        return x

    out_file = tmpdir.join("huge.onnx")
    t = Tensor(shape=(1,), dtype=DType.FLOAT32, name="x")
    with patch.object(onnx_pb2.ModelProto, "ByteSize", return_value=3 * 1024 * 1024 * 1024):
        export(simple_func, t, str(out_file))
    assert out_file.exists()


def test_exporter_file_like_object() -> None:
    """Tests the test_exporter_file_like_object functionality."""
    import io

    from onnx9000.converters.frontend.exporter import export
    from onnx9000.converters.frontend.tensor import Tensor

    def simple_func(x):
        """Test the simple_func functionality."""
        return x

    t = Tensor(shape=(1,), dtype=DType.FLOAT32, name="x")
    f = io.BytesIO()
    export(simple_func, t, f)
    assert len(f.getvalue()) > 0


def test_exporter_visualize() -> None:
    """Tests the test_exporter_visualize functionality."""
    from onnx9000.converters.frontend.exporter import visualize

    assert visualize("dummy.onnx") == ""
