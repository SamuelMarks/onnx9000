"""Tests for Caffe prototxt parser."""

from onnx9000.converters.caffe.parser import parse_prototxt


def test_parse_prototxt_basic():
    """Test parsing basic prototxt."""
    content = """
    name: "LeNet"
    layer {
      name: "mnist"
      type: "Data"
      top: "data"
      top: "label"
      data_param {
        source: "mnist_train_lmdb"
        batch_size: 64
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 20
        kernel_size: 5
      }
    }
    """

    net = parse_prototxt(content)
    assert net["name"][0] == "LeNet"
    assert len(net["layer"]) == 2

    l0 = net["layer"][0]
    assert l0["name"][0] == "mnist"
    assert l0["type"][0] == "Data"
    assert l0["top"] == ["data", "label"]
    assert l0["data_param"][0]["batch_size"][0] == 64

    l1 = net["layer"][1]
    assert l1["name"][0] == "conv1"
    assert l1["type"][0] == "Convolution"
    assert l1["convolution_param"][0]["num_output"][0] == 20
