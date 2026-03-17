from onnx9000.converters.frontend.nn.functional import interpolate
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_interpolate_coverage() -> None:
    t = Tensor(shape=(1, 3, 10, 10), dtype=DType.FLOAT32, name="t")
    assert interpolate(t, mode="linear", align_corners=True) is None
    assert interpolate(t, scale_factor=2.0, size=(20, 20)) is None
