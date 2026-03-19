"""Provide functionality for this module."""

from onnx import onnx_pb as _onnx_pb
import sys

sys.modules[__name__] = _onnx_pb
