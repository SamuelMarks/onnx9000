"""WebGPU Backend package for ONNX9000."""

from onnx9000.backends.webgpu.executor import WebGPUExecutionProvider
from onnx9000.backends.webgpu.wgsl import WGSLGenerator

__all__ = ["WGSLGenerator", "WebGPUExecutionProvider"]
