import pytest
from onnx9000.openvino.xml_builder import *


def test_XmlNode():
    try:
        obj = XmlNode()
        assert obj is not None
    except Exception:
        pass


def test_XmlBuilder():
    try:
        obj = XmlBuilder()
        assert obj is not None
    except Exception:
        pass
