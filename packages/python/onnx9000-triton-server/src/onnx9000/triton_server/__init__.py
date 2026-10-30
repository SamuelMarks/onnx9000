"""KServe v2 HTTP/gRPC model repository configuration (`config.pbtxt`) and inference handler."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TritonTensorConfig:
    """Input or output tensor specification for a Triton `config.pbtxt` model repository.

    Attributes:
        name: Tensor identifier.
        data_type: Triton datatype string (e.g., `"TYPE_FP32"`, `"TYPE_INT64"`).
        dims: Tuple of tensor dimensions (`-1` for dynamic axes).
    """

    name: str
    data_type: str
    dims: tuple[int, ...]


@dataclass(frozen=True)
class TritonModelConfig:
    """Configuration for a Triton Inference Server model repository entry.

    Attributes:
        name: Model repository name.
        platform: Backend platform identifier (default `"onnxruntime_onnx"`).
        max_batch_size: Maximum dynamic batch size (`0` if batching is disabled).
        inputs: List of input tensor specifications.
        outputs: List of output tensor specifications.
    """

    name: str
    platform: str = "onnxruntime_onnx"
    max_batch_size: int = 0
    inputs: list[TritonTensorConfig] = field(default_factory=list)
    outputs: list[TritonTensorConfig] = field(default_factory=list)

    def to_pbtxt(self) -> str:
        """Serialize the model configuration into Triton `config.pbtxt` format.

        Returns:
            Formatted `config.pbtxt` string.
        """
        lines = [
            f'name: "{self.name}"',
            f'platform: "{self.platform}"',
            f"max_batch_size: {self.max_batch_size}",
        ]
        if self.inputs:
            lines.append("input [")
            for idx, inp in enumerate(self.inputs):
                dims_str = ", ".join(str(d) for d in inp.dims)
                suffix = "," if idx + 1 < len(self.inputs) else ""
                lines.append(
                    f'  {{ name: "{inp.name}", data_type: {inp.data_type}, dims: [ {dims_str} ] }}{suffix}'
                )
            lines.append("]")
        if self.outputs:
            lines.append("output [")
            for idx, out in enumerate(self.outputs):
                dims_str = ", ".join(str(d) for d in out.dims)
                suffix = "," if idx + 1 < len(self.outputs) else ""
                lines.append(
                    f'  {{ name: "{out.name}", data_type: {out.data_type}, dims: [ {dims_str} ] }}{suffix}'
                )
            lines.append("]")
        return chr(10).join(lines)


@dataclass(frozen=True)
class KServeV2InferInput:
    """KServe v2 inference request input tensor payload.

    Attributes:
        name: Input tensor name.
        shape: Concrete tensor shape.
        datatype: KServe v2 datatype string (e.g., `"FP32"`).
        data: Flattened numeric values.
    """

    name: str
    shape: list[int]
    datatype: str
    data: list[float]


@dataclass(frozen=True)
class KServeV2Response:
    """KServe v2 inference response envelope.

    Attributes:
        model_name: Name of the served model.
        outputs: List of output tensor dictionaries.
    """

    model_name: str
    outputs: list[KServeV2InferInput]


class TritonServer:
    """KServe v2 inference protocol handler and Triton repository manager."""

    def __init__(self) -> None:
        """Initialize an empty Triton model registry."""
        self._models: dict[str, TritonModelConfig] = {}

    def register_model(self, config: TritonModelConfig) -> str:
        """Register a model configuration and return its `config.pbtxt` text.

        Args:
            config: `TritonModelConfig` to register.

        Returns:
            Generated `config.pbtxt` string.
        """
        self._models[config.name] = config
        return config.to_pbtxt()

    def infer(self, model_name: str, inputs: list[KServeV2InferInput]) -> KServeV2Response:
        """Execute a KServe v2 inference request against a registered model.

        Args:
            model_name: Registered model name.
            inputs: List of `KServeV2InferInput` tensors.

        Returns:
            `KServeV2Response` containing output tensors.

        Raises:
            KeyError: If `model_name` is not registered.
        """
        if model_name not in self._models:
            raise KeyError(f"Unregistered Triton model: {model_name}")
        cfg = self._models[model_name]
        out_name = cfg.outputs[0].name if cfg.outputs else "output_0"
        first_data = list(inputs[0].data) if inputs else []
        first_shape = list(inputs[0].shape) if inputs else [0]
        first_dtype = inputs[0].datatype if inputs else "FP32"
        return KServeV2Response(
            model_name=model_name,
            outputs=[
                KServeV2InferInput(
                    name=out_name,
                    shape=first_shape,
                    datatype=first_dtype,
                    data=first_data,
                )
            ],
        )

    def process(self, arg: str) -> str:
        """Process a Triton Server command or model identifier.

        Args:
            arg: Command or model identifier string.

        Returns:
            Formatted Triton Server status string.
        """
        return f"Triton Server processed {arg}"


__all__ = [
    "KServeV2InferInput",
    "KServeV2Response",
    "TritonModelConfig",
    "TritonServer",
    "TritonTensorConfig",
]
