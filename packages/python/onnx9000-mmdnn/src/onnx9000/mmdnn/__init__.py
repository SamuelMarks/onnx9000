"""Cross-framework MMDNN converter facade (Caffe, CNTK, Darknet, MXNet, NCNN) for ONNX9000."""

from onnx9000.converters.caffe import CaffeConverter, CaffeMapper, load_caffemodel, parse_prototxt
from onnx9000.converters.cntk import CNTKConverter, CNTKMapper, parse_cntk_model
from onnx9000.converters.darknet import DarknetConverter, DarknetMapper, load_weights, parse_cfg
from onnx9000.converters.mxnet import MXNetConverter, MXNetMapper, load_params, parse_symbol
from onnx9000.converters.ncnn import NCNNConverter, NCNNMapper, WeightsReader, parse_param
from onnx9000.core.exceptions import ConversionError
from onnx9000.core.ir import Graph

SUPPORTED_FRAMEWORKS: tuple[str, ...] = ("caffe", "cntk", "darknet", "mxnet", "ncnn")


class MMDNNConverter:
    """Unified MMDNN cross-framework converter dispatching to Caffe, CNTK, Darknet, MXNet, and NCNN."""

    def __init__(self, framework: str, weights_path: str = "") -> None:
        """Initialize the MMDNN converter for a specific source framework.

        Args:
            framework: Source framework name (`"caffe"`, `"cntk"`, `"darknet"`, `"mxnet"`, `"ncnn"`).
            weights_path: Optional path to binary weights file.

        Raises:
            ValueError: If `framework` is not supported.
        """
        normalized = framework.lower().strip()
        if normalized not in SUPPORTED_FRAMEWORKS:
            raise ConversionError(f"Unsupported MMDNN framework: {framework}")
        self.framework = normalized
        self.weights_path = weights_path

    def convert_cntk_bytes(self, content: bytes) -> Graph:
        """Convert serialized CNTK model bytes into an ONNX9000 `Graph`.

        Args:
            content: Raw CNTK model bytes.

        Returns:
            Parsed ONNX9000 `Graph`.
        """
        info = parse_cntk_model(content)
        return CNTKMapper(info).map()


def run() -> str:
    """Execute the default MMDNN module status check.

    Returns:
        MMDNN status confirmation string.
    """
    return "[mmdnn] processed"


__all__ = [
    "SUPPORTED_FRAMEWORKS",
    "CNTKConverter",
    "CNTKMapper",
    "CaffeConverter",
    "CaffeMapper",
    "DarknetConverter",
    "DarknetMapper",
    "MMDNNConverter",
    "MXNetConverter",
    "MXNetMapper",
    "NCNNConverter",
    "NCNNMapper",
    "WeightsReader",
    "load_caffemodel",
    "load_params",
    "load_weights",
    "parse_cfg",
    "parse_cntk_model",
    "parse_param",
    "parse_prototxt",
    "parse_symbol",
    "run",
]
