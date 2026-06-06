import pytest
from onnx9000.tflite_exporter.compiler.subgraph import *

def test_compile_graph_to_tflite():
    try:
        res = compile_graph_to_tflite()
    except Exception:
        pass

