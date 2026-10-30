"""Exhaustive 100% coverage test suite for onnx9000-converters."""

from __future__ import annotations

import ast
import io
import struct
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from onnx9000.converters import caffe, cntk, jit, ncnn, onnx_parser
from onnx9000.converters.caffe.mapper import CaffeMapper
from onnx9000.converters.caffe.parser import parse_prototxt
from onnx9000.converters.caffe.weights import ProtobufDecoder, load_caffemodel, parse_blob
from onnx9000.converters.cntk.mapper import CNTKMapper
from onnx9000.converters.darknet.mapper import DarknetMapper
from onnx9000.converters.darknet.parser import parse_cfg
from onnx9000.converters.frontend import (
    ast_parser,
    bonsai,
)
from onnx9000.converters.frontend import (
    builder as fe_builder,
)
from onnx9000.converters.frontend import (
    exporter as fe_exporter,
)
from onnx9000.converters.frontend import (
    jit as fe_jit,
)
from onnx9000.converters.frontend import (
    quantization as fe_quant,
)
from onnx9000.converters.frontend import (
    tracer as fe_tracer,
)
from onnx9000.converters.frontend import (
    utils as fe_utils,
)
from onnx9000.converters.frontend import (
    weight_utils as fe_wutils,
)
from onnx9000.converters.frontend.nn import (
    conv as fe_conv,
)
from onnx9000.converters.frontend.nn import (
    functional as fe_F,
)
from onnx9000.converters.frontend.nn import (
    init as fe_init,
)
from onnx9000.converters.frontend.nn import (
    module as fe_module,
)
from onnx9000.converters.frontend.nn import (
    normalization as fe_norm,
)
from onnx9000.converters.frontend.nn import (
    rnn as fe_rnn,
)
from onnx9000.converters.frontend.tensor import (
    Node as FENode,
)
from onnx9000.converters.frontend.tensor import (
    Tensor as FETensor,
)
from onnx9000.converters.mltools import libsvm, sparkml
from onnx9000.converters.mxnet.mapper import MXNetMapper
from onnx9000.converters.mxnet.weights import load_params
from onnx9000.converters.ncnn.mapper import NCNNMapper
from onnx9000.converters.ncnn.parser import parse_param
from onnx9000.converters.ncnn.weights import WeightsReader
from onnx9000.converters.paddle import parsers as paddle_parsers
from onnx9000.converters.sklearn import builder as skl_builder
from onnx9000.converters.tf import (
    api as tf_api,
)
from onnx9000.converters.tf import (
    builder as tf_builder,
)
from onnx9000.converters.tf import (
    keras_h5_parser,
    keras_v3_parser,
)
from onnx9000.converters.tf import (
    nn_ops as tf_nn,
)
from onnx9000.converters.tf import (
    parsers as tf_parsers,
)
from onnx9000.converters.tf import (
    passes as tf_passes,
)
from onnx9000.converters.torch import fx as torch_fx
from onnx9000.converters.torch import script as torch_script
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ConversionError
from onnx9000.core.ir import (
    Constant,
    Graph,
    Node,
    Tensor,
    ValueInfo,
    Variable,
)


def test_caffe_cntk_darknet_mxnet_ncnn() -> None:
    """Cover all lines and branches in caffe, cntk, darknet, mxnet, and ncnn converters."""
    # Caffe parser
    proto_str = """
    name: "TestNet"
    # comment
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 2
        kernel_size: 3
        stride: 1
        pad: 1
        bias_term: true
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
        pad: 0
      }
    }
    layer {
      name: "pool_avg"
      type: "Pooling"
      bottom: "pool1"
      top: "pool_avg"
      pooling_param {
        pool: 1
        kernel_size: 2
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "pool_avg"
      top: "relu1"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "relu1"
      top: "ip1"
      inner_product_param {
        num_output: 4
      }
    }
    layer {
      name: "sm1"
      type: "Softmax"
      bottom: "ip1"
      top: "sm1"
      softmax_param {
        axis: 1
      }
    }
    layer {
      name: "do1"
      type: "Dropout"
      bottom: "sm1"
      top: "do1"
      dropout_param {
        dropout_ratio: 0.5
      }
    }
    layer {
      name: "bn1"
      type: "BatchNorm"
      bottom: "do1"
      top: "bn1"
      batch_norm_param {
        eps: 0.001
      }
    }
    layer {
      name: "sc1"
      type: "Scale"
      bottom: "bn1"
      top: "sc1"
    }
    layer {
      name: "elt_sum"
      type: "Eltwise"
      bottom: "sc1"
      bottom: "bn1"
      top: "elt_sum"
      eltwise_param {
        operation: SUM
      }
    }
    layer {
      name: "elt_prod"
      type: "Eltwise"
      bottom: "elt_sum"
      bottom: "bn1"
      top: "elt_prod"
      eltwise_param {
        operation: 0
      }
    }
    layer {
      name: "elt_max"
      type: "Eltwise"
      bottom: "elt_prod"
      bottom: "bn1"
      top: "elt_max"
      eltwise_param {
        operation: MAX
      }
    }
    layer {
      name: "cat1"
      type: "Concat"
      bottom: "elt_max"
      bottom: "bn1"
      top: "cat1"
      concat_param {
        axis: 1
      }
    }
    layer {
      name: "unk1"
      type: "CustomCaffe"
      bottom: "cat1"
      top: "unk1"
    }
    """
    parsed_proto = parse_prototxt(proto_str)
    assert parsed_proto["name"] == ["TestNet"]

    # Caffe weights ProtobufDecoder
    # Build a synthetic caffemodel binary containing field 2 (layer) and field 100 (v1 layer)
    # BlobProto: field 1 (num=1), field 2 (channels=1), field 3 (height=1), field 4 (width=2),
    # field 5 (data: packed float or single float), field 6 (diff: skip), field 7 (shape: dim=2), field 8 (double_data)
    def encode_varint(v: int) -> bytes:
        """Test helper function encode_varint."""
        out = bytearray()
        while v > 0x7F:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
        return bytes(out)

    # Blob1: 2D shape (2, 2) via field 7 (wire 2), data via field 5 (wire 2 packed floats + wire 5 unpacked floats = 4 floats)
    shape_msg = (
        b"\x08\x02" + b"\x0a\x01\x02" + b"\x18\x00"
    )  # dim=2 (wire 0 and wire 2) + field 3 skip
    blob1 = (
        b"\x3a"
        + encode_varint(len(shape_msg))
        + shape_msg
        + b"\x2a\x08"
        + struct.pack("<2f", 1.0, 2.0)  # field 5 wire 2
        + b"\x2d"
        + struct.pack("<f", 3.0)  # field 5 wire 5
        + b"\x2d"
        + struct.pack("<f", 4.0)  # field 5 wire 5
        + b"\x30\x00"  # field 6 skip
    )
    # Blob2: legacy num=1, channels=1, height=1, width=2 (2 floats)
    blob2 = (
        b"\x08\x01\x10\x01\x18\x01\x20\x02"
        + b"\x2a\x08"
        + struct.pack("<2f", 1.0, 2.0)
        + b"\x42\x08"
        + struct.pack("<d", 4.0)  # field 8 wire 2 skip
        + b"\x41"
        + struct.pack("<d", 5.0)  # field 8 wire 1 skip
    )
    # Layer 1 (conv1): field 1 (name="conv1"), field 6 (blob1), field 6 (blob2), field 99 (skip wire 1)
    layer1 = (
        b"\x0a\x05conv1"
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
        + b"\x32"
        + encode_varint(len(blob2))
        + blob2
        + b"\x99\x06"
        + b"\x00" * 8
    )
    # Layer 2 (bn1): 3 blobs for BatchNorm
    layer2 = (
        b"\x0a\x03bn1"
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
    )
    # Layer 3 (sc1): 2 blobs for Scale
    layer3 = (
        b"\x0a\x03sc1"
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
    )
    # Layer 4 (ip1): 2 blobs for InnerProduct
    layer4 = (
        b"\x0a\x03ip1"
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
        + b"\x32"
        + encode_varint(len(blob1))
        + blob1
    )
    caffemodel_bytes = (
        b"\x12"
        + encode_varint(len(layer1))
        + layer1
        + b"\x92\x06"
        + encode_varint(len(layer2))
        + layer2
        + b"\x12"
        + encode_varint(len(layer3))
        + layer3
        + b"\x12"
        + encode_varint(len(layer4))
        + layer4
        + b"\x08\x01"  # top-level field 1 skip
    )
    weights = load_caffemodel(io.BytesIO(caffemodel_bytes))
    assert "conv1" in weights

    # Test ProtobufDecoder methods directly (read_bytes, skip errors, etc.)
    dec = ProtobufDecoder(b"ab\x00\x00\x00\x00")
    assert dec.read_bytes(2) == b"ab"
    dec.skip(5)
    assert dec.read_tag() == (0, 0)
    assert parse_blob(b"\x00") is not None
    assert load_caffemodel(io.BytesIO(b"\x00")) == {}
    from onnx9000.converters.caffe.weights import parse_layer

    assert parse_layer(b"\x00")["name"] == ""
    with pytest.raises(EOFError):
        dec.read_varint()
    dec2 = ProtobufDecoder(b"\x80\x80")
    with pytest.raises(EOFError):
        dec2.read_varint()
    with pytest.raises(ConversionError):
        dec2.skip(99)

    mapper = CaffeMapper(parsed_proto, weights)
    g_caffe = mapper.map()
    assert len(g_caffe.nodes) > 0

    # Also test single layer dict (not list) in CaffeMapper
    single_proto = parse_prototxt(
        'name: "Single"\ninput: "x"\nlayer {\n  name: "r"\n  type: "ReLU"\n  bottom: "x"\n  top: "y"\n}\n'
    )
    CaffeMapper(single_proto, {}).map()

    with tempfile.TemporaryDirectory() as tmpdir:
        p_file = Path(tmpdir) / "net.prototxt"
        w_file = Path(tmpdir) / "net.caffemodel"
        p_file.write_text(proto_str)
        w_file.write_bytes(caffemodel_bytes)
        conv = caffe.CaffeConverter(str(w_file))
        assert len(conv.parse(str(p_file)).nodes) > 0
        assert len(conv.parse(proto_str).nodes) > 0

    # CNTK
    with tempfile.TemporaryDirectory() as tmpdir:
        cntk_file = Path(tmpdir) / "m.model"
        cntk_file.write_bytes(b"\x08\x01\x10\x02")
        cntk_conv = cntk.CNTKConverter()
        assert cntk_conv.parse(str(cntk_file)) is not None
        assert cntk_conv.parse("raw_string_model") is not None
        assert cntk_conv.parse(b"\x08\x01") is not None  # type: ignore[arg-type]

    cntk_info = {
        "inputs": [{"name": "x"}],
        "outputs": [{"name": "y"}],
        "nodes": [
            {"op": "Convolution", "name": "c", "inputs": ["x"], "outputs": ["h1"]},
            {"op": "Plus", "name": "p", "inputs": ["h1", "x"], "outputs": ["h2"]},
            {"op": "ElementTimes", "name": "m", "inputs": ["h2", "x"], "outputs": ["h3"]},
            {"op": "Times", "name": "t", "inputs": ["h3", "x"], "outputs": ["h4"]},
            {"op": "RectifiedLinear", "name": "r", "inputs": ["h4"], "outputs": ["h5"]},
            {"op": "CustomCntk", "name": "u", "inputs": ["h5"], "outputs": ["y"]},
        ],
    }
    assert len(CNTKMapper(cntk_info).map().nodes) == 6

    # Darknet
    cfg_str = """
    key_before_block=1
    [net]
    width=32
    height=32
    channels=3

    [convolutional]
    batch_normalize=1
    filters=2
    size=3
    stride=1
    pad=1
    activation=mish

    [convolutional]
    batch_normalize=0
    filters=2
    size=1
    stride=1
    pad=0
    activation=swish

    [convolutional]
    filters=2
    size=1
    stride=1
    pad=0
    activation=logistic

    [route]
    layers=-1, -2

    [shortcut]
    from=-2
    activation=leaky

    [maxpool]
    size=2
    stride=2

    [upsample]
    stride=2

    [yolo]
    """
    blocks = parse_cfg(cfg_str)
    dm = DarknetMapper(blocks, np.zeros(1000, dtype=np.float32))
    assert len(dm.map().nodes) > 0

    # MXNet
    with tempfile.TemporaryDirectory() as tmpdir:
        p_bin = Path(tmpdir) / "m.params"
        p_bin.write_bytes(b"")
        with open(p_bin, "rb") as f:
            assert load_params(f) == {}
        p_bin.write_bytes(b"12345678")
        with open(p_bin, "rb") as f:
            assert load_params(f) == {}

    # NCNN
    ncnn_param_str = """7767517
    12 13
    Input in 0 1 data 0=3 1=32 2=32
    Convolution conv1 1 1 data c1 0=2 1=1 11=1 2=1 12=1 3=1 13=1 4=0 5=1 6=2
    ConvolutionDepthWise dw1 1 1 c1 c2 0=2 1=1 2=1 3=1 4=0 5=0 6=2 7=2
    BatchNorm bn1 1 1 c2 c3 0=2
    ReLU relu1 1 1 c3 c4 0=0.0
    ReLU lrelu1 1 1 c4 c5 0=0.1
    Pooling pool_max 1 1 c5 c6 0=0 1=2 2=2 3=0 4=0
    Pooling pool_avg 1 1 c6 c7 0=1 1=2 2=2 3=0 4=1
    Pooling pool_gavg 1 1 c7 c8 0=1 1=2 2=2 3=0 4=1
    InnerProduct ip1 1 1 c8 c9 0=2 1=1 2=4
    Concat cat1 2 1 c8 c9 c10 0=1
    BinaryOp bin_add 2 1 c10 c9 c11 0=0
    BinaryOp bin_sub 2 1 c11 c9 c12 0=1
    BinaryOp bin_mul 2 1 c12 c9 c13 0=2
    BinaryOp bin_div 2 1 c13 c9 c14 0=3
    Eltwise elt_prod 2 1 c14 c9 c15 0=0
    Eltwise elt_sum 2 1 c15 c9 c16 0=1
    Eltwise elt_max 2 1 c16 c9 c17 0=2
    Softmax sm1 1 1 c17 c18 0=1
    Split split1 1 2 c18 s1 s2
    CustomNcnn cust1 1 2 s1 out out 0=1,2,3
    """
    ncnn_info = parse_param(ncnn_param_str)
    NCNNMapper({"layers": []}, WeightsReader(io.BytesIO(b""))).map()
    # Modify pool_gavg to have global_pooling=1 (param 4=1)
    assert len(ncnn_info["layers"]) > 0
    with pytest.raises(ConversionError):
        parse_param("")
    with pytest.raises(ConversionError):
        parse_param("""12345
1 1
""")

    # Test WeightsReader magic tags: fp32 (0x00000000), fp16 (0x01306B47), int8 (0x000D4B38), raw floats
    w_bytes = (
        struct.pack("<I", 0)
        + struct.pack("<2f", 1.0, 2.0)  # fp32 tag + 2 floats
        + struct.pack("<2f", 0.5, 0.5)  # raw floats for conv1 bias
        + struct.pack("<I", 0x01306B47)
        + np.array([1.0, 2.0], dtype=np.float16).tobytes()  # fp16 tag
        + struct.pack("<8f", *([1.0] * 8))  # bn1 4*2 floats
        + struct.pack("<I", 0x000D4B38)
        + b"\x01\x02\x03\x04"  # int8 tag + 4 bytes for ip1
        + struct.pack("<2f", 0.1, 0.2)  # ip1 bias
    )
    wr = WeightsReader(io.BytesIO(w_bytes))
    assert len(wr.read_blob(0)) == 0
    assert len(WeightsReader(io.BytesIO(b"")).read_blob(2)) == 2
    assert len(NCNNMapper(ncnn_info, WeightsReader(io.BytesIO(w_bytes))).map().nodes) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        p_path = Path(tmpdir) / "m.param"
        b_path = Path(tmpdir) / "m.bin"
        p_path.write_text(ncnn_param_str)
        b_path.write_bytes(w_bytes)
        assert len(ncnn.NCNNConverter(str(b_path)).parse(str(p_path)).nodes) > 0
        assert len(ncnn.NCNNConverter(str(b_path)).parse(ncnn_param_str).nodes) > 0

    # Additional Caffe / Darknet / CNTK / NCNN edge cases
    # Caffe top-level input_dim, input_shape, Input layer, Data layer, layers key
    proto_inputs_str = """
    name: "InNet"
    input: "in1"
    input: "in2"
    input_shape {
      dim: 1
      dim: 3
    }
    layers {
      name: "inp_layer"
      type: "Input"
      top: "in1"
      top: "in3"
      input_param {
        shape {
          dim: 1
          dim: 3
        }
      }
    }
    layers {
      name: "data_layer"
      type: "Data"
      top: "in1"
      top: "in4"
    }
    layers {
      name: "empty_unk"
      type: "EmptyUnk"
    }
    layers {
      name: "r1"
      type: "ReLU"
      bottom: "in1"
      top: "shared_out"
    }
    layers {
      name: "r2"
      type: "ReLU"
      bottom: "in2"
      top: "shared_out"
    }
    """
    CaffeMapper(parse_prototxt(proto_inputs_str), {}).map()

    proto_legacy_dims = parse_prototxt(
        'name: "Legacy"\ninput: "in1"\ninput_dim: 1\ninput_dim: 3\ninput_dim: 224\ninput_dim: 224\ninput_dim: 0\nlayer {\n  name: "t"\n  type: "ReLU"\n  bottom: "in1"\n  top: "out"\n  bias_term: true\n}\n}'
    )
    CaffeMapper(proto_legacy_dims, {}).map()
    # CaffeMapper with input_names and empty input_dims/input_shapes
    CaffeMapper({"input": ["in_no_shape"], "layer": []}, {}).map()

    # Caffe weights: parse_blob with legacy dims where size != n*c*h*w, and layer with empty name/blobs
    blob_mismatch = b"\x08\x02\x2d" + struct.pack("<f", 1.0)
    layer_empty = b"\x08\x01"
    caffemodel_edge = (
        b"\x12"
        + encode_varint(len(layer_empty))
        + layer_empty
        + b"\x12"
        + encode_varint(len(b"\x0a\x02l1\x32" + encode_varint(len(blob_mismatch)) + blob_mismatch))
        + b"\x0a\x02l1\x32"
        + encode_varint(len(blob_mismatch))
        + blob_mismatch
    )
    load_caffemodel(io.BytesIO(caffemodel_edge))

    # CNTKMapper with duplicate output
    CNTKMapper({"inputs": [], "outputs": [{"name": "y"}, {"name": "y"}], "nodes": []}).map()

    # Darknet edge cases: empty line, comment line, net at end, leaky conv, linear conv, route 1 layer, shortcut linear, maxpool stride=1, existing output
    cfg_edge = """
    # comment

    [convolutional]
    filters=2
    size=1
    stride=1
    pad=0
    activation=leaky

    [convolutional]
    filters=2
    size=1
    stride=1
    pad=0
    activation=linear

    [route]
    layers=-1

    [shortcut]
    from=0
    activation=linear

    [maxpool]
    size=2
    stride=1

    [net]
    channels=3
    """
    dm_edge = DarknetMapper(parse_cfg(cfg_edge), np.zeros(100, dtype=np.float32))
    # Pre-populate output to hit 221->224 if needed, or run map() twice
    dm_edge.map()
    dm_edge.map()


def test_onnx_parser_exhaustive() -> None:
    """Cover 100% of onnx_parser.py."""

    def encode_varint(v: int) -> bytes:
        """Test helper function encode_varint."""
        out = bytearray()
        while v > 0x7F:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
        return bytes(out)

    # Build a synthetic ONNX ModelProto binary covering all wire types and fields
    # TensorProto: field 1 packed (wire 2), field 1 unpacked (wire 0), field 2 (data_type), field 8 (name), field 9 (raw_data), field 99 (skip wire 5)
    tp_bytes = (
        b"\x0a\x02\x02\x03"  # field 1 packed dims [2, 3]
        + b"\x08\x04"  # field 1 unpacked dim 4
        + b"\x08\x05"  # field 1 unpacked dim 5 (when "dims" already in result)
        + b"\x0a\x01\x06"  # field 1 packed dim [6] (when "dims" already in result)
        + b"\x10\x01"  # field 2 data_type = 1
        + b"\x42\x02w1"  # field 8 name = "w1"
        + b"\x4a\x041234"  # field 9 raw_data
        + b"\x1d1234"  # field 3 wire 5 (FIXED32) skip
    )
    # AttributeProto: skip field wire 0, wire 1 (FIXED64), wire 2 (LENGTH_DELIMITED)
    attr_bytes = b"\x08\x01"
    # NodeProto: field 1 (input x2), field 2 (output x2), field 3 (name), field 4 (op_type), field 5 (attribute x2), field 6 (skip wire 1), field 7 (skip wire 2)
    node_bytes = (
        b"\x0a\x02in"
        + b"\x0a\x03in2"
        + b"\x12\x03out"
        + b"\x12\x04out2"
        + b"\x1a\x02n1"
        + b"\x22\x04Relu"
        + b"\x2a"
        + encode_varint(len(attr_bytes))
        + attr_bytes
        + b"\x2a"
        + encode_varint(len(attr_bytes))
        + attr_bytes
        + b"\x31"
        + b"12345678"
        + b"\x3a\x02ab"
        + b"\x40\x01"
    )
    vi_bytes = b"\x0a\x02vi"
    # GraphProto: field 1 (node x2), field 2 (name), field 5 (initializer x2), field 11 (input x2), field 12 (output x2), field 99 (skip)
    graph_bytes = (
        b"\x0a"
        + encode_varint(len(node_bytes))
        + node_bytes
        + b"\x0a"
        + encode_varint(len(node_bytes))
        + node_bytes
        + b"\x12\x02g1"
        + b"\x2a"
        + encode_varint(len(tp_bytes))
        + tp_bytes
        + b"\x2a"
        + encode_varint(len(tp_bytes))
        + tp_bytes
        + b"\x5a"
        + encode_varint(len(vi_bytes))
        + vi_bytes
        + b"\x5a"
        + encode_varint(len(vi_bytes))
        + vi_bytes
        + b"\x62"
        + encode_varint(len(vi_bytes))
        + vi_bytes
        + b"\x62"
        + encode_varint(len(vi_bytes))
        + vi_bytes
        + b"\x48\x01"
    )
    # ModelProto: field 7 (graph), field 1 (ir_version=128 -> multi-byte varint)
    model_bytes = b"\x08\x80\x01" + b"\x3a" + encode_varint(len(graph_bytes)) + graph_bytes

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test.onnx"
        onnx_file.write_bytes(model_bytes)
        with onnx_parser.PureOnnxParser(str(onnx_file)) as parser:
            res = parser.parse_model()
            assert res["graph"]["name"] == "g1"
            # Test BufferError handling on __exit__ by keeping a reference or mocking close
        p2 = onnx_parser.PureOnnxParser(str(onnx_file))
        p2.__exit__(None, None, None)
        p2.__exit__(None, None, None)

        class BrokenMmap:
            """Test helper class BrokenMmap."""

            def close(self) -> None:
                """Test helper function close."""
                raise BufferError("exported pointers")

        p3 = onnx_parser.PureOnnxParser(str(onnx_file))
        del p3.view
        p3.mm.close()
        p3.mm = BrokenMmap()  # type: ignore[assignment]
        p3.__exit__(None, None, None)

    with pytest.raises(EOFError):
        onnx_parser.read_varint(memoryview(b"\x80"), 0)
    with pytest.raises(ConversionError):
        onnx_parser.skip_field(memoryview(b"\x00"), 0, 3)


def test_tf_passes_and_keras_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover 100% of tf/passes.py, tf/keras_v3_parser.py, and tf/keras_h5_parser.py."""
    # tf/passes.py
    g = Graph("tf_opt")
    g.outputs.extend(
        [ValueInfo("id_out", (1,), DType.FLOAT32), ValueInfo("final_out", (1,), DType.FLOAT32)]
    )
    g.nodes.extend(
        [
            Node("Identity", ["x"], ["id_out"]),
            Node("Dropout", ["id_out"], ["do_out"]),
            Node("Custom_TFStopGradient", ["do_out"], ["sg_out"]),
            Node("Custom_TFPrint", ["sg_out"], ["pr_out"]),
            Node("Transpose", ["sg_out"], ["tr1"], attributes={"perm": [1, 0]}),
            Node("Transpose", ["tr1"], ["tr2"], attributes={"perm": [1, 0]}),
            Node("Transpose", ["tr2"], ["tr3"], attributes={"perm": [0, 1]}),
            Node("Transpose", ["tr3"], ["tr4"], attributes={"perm": [1, 0]}),
            Node("Relu", ["tr2"], ["final_out"]),
            Node("Relu", ["unused_in"], ["unused_out"]),
        ]
    )
    tf_passes.tf_optimize_graph(g)

    # Also test ImportError branches in constant_folding_pass, shape_folding_pass, pattern_matching_pass
    orig_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        """Test helper function fake_import."""
        if name.startswith("onnx9000.optimizer.simplifier"):
            raise ImportError("mocked")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    tf_passes.constant_folding_pass(g)
    tf_passes.shape_folding_pass(g)
    tf_passes.pattern_matching_pass(g)
    monkeypatch.undo()

    # tf/keras_v3_parser.py
    class FakeKerasTensor:
        """Test helper class FakeKerasTensor."""

        def __init__(self, name: str | None) -> None:
            """Test helper function __init__."""
            self.name = name

    class FakeWeight:
        """Test helper class FakeWeight."""

        def __init__(self, name: str, arr: np.ndarray) -> None:
            """Test helper function __init__."""
            self.name = name
            self._arr = arr
            self.dtype = "float32"

        def numpy(self) -> np.ndarray:
            """Test helper function numpy."""
            return self._arr

    class FakeInboundNode:
        """Test helper class FakeInboundNode."""

        def __init__(self, ins: list[FakeKerasTensor], outs: list[FakeKerasTensor]) -> None:
            """Test helper function __init__."""
            self.input_tensors = ins
            self.output_tensors = outs

    class FakeOp:
        """Test helper class FakeOp."""

        def __init__(
            self,
            name: str,
            cls_name: str,
            mod_name: str,
            nodes: list[FakeInboundNode],
            weights: list[FakeWeight] | None = None,
            cfg: dict[str, object] | None = None,
        ) -> None:
            """Test helper function __init__."""
            self.name = name
            cfg_dict = cfg or {}
            cls = type(cls_name, (), {"__module__": mod_name, "get_config": lambda self: cfg_dict})
            self.__class__ = cls
            self._inbound_nodes = nodes
            self.weights = weights or []

    kt_in = FakeKerasTensor("keras_tensor_0")
    kt_out1 = FakeKerasTensor("kt_out1")
    kt_out2 = FakeKerasTensor(None)

    class FakeKeras3Model:
        """Test helper class FakeKeras3Model."""

        built = True
        operations = [
            FakeOp("in_op", "InputLayer", "keras.layers", [FakeInboundNode([kt_in], [kt_in])]),
            FakeOp(
                "dense_op",
                "Dense",
                "custom.layers",
                [
                    FakeInboundNode([kt_in], [kt_out1]),
                    FakeInboundNode([kt_in], [kt_out2]),
                ],
                [FakeWeight("w1", np.ones((2, 2), dtype=np.float32))],
                {"units": 4, "_private": 1},
            ),
        ]

    k3_parser = keras_v3_parser.Keras3Parser(FakeKeras3Model())
    tf_g = k3_parser.parse()
    assert len(tf_g.nodes) > 0

    # Test subclassed model building with single input_shape and list of tuple input_shapes, plus missing input_shape error
    class FakeKerasModelCls:
        """Test helper class FakeKerasModelCls."""

        def __init__(self, *_a: object, **_kw: object) -> None:
            """Test helper function __init__."""
            self.built = True
            self.operations = FakeKeras3Model.operations

    fake_keras_mod = types.ModuleType("keras")
    fake_keras_mod.Input = lambda batch_shape: FakeKerasTensor("built_in")  # type: ignore[attr-defined]
    fake_keras_mod.Model = FakeKerasModelCls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keras", fake_keras_mod)

    class SubclassedModel:
        """Test helper class SubclassedModel."""

        built = False

        def __call__(self, x: object) -> object:
            """Test helper function __call__."""
            return x

    assert (
        len(keras_v3_parser.Keras3Parser(SubclassedModel(), input_shape=(1, 4)).parse().nodes) > 0
    )
    assert (
        len(
            keras_v3_parser.Keras3Parser(SubclassedModel(), input_shape=[(1, 4), (1, 2)])
            .parse()
            .nodes
        )
        > 0
    )

    with pytest.raises(ConversionError):
        keras_v3_parser.Keras3Parser(SubclassedModel()).parse()

    # Test convert_keras_to_onnx with is_v3=True and is_v3=False
    assert tf_api.convert_keras_to_onnx(FakeKerasModelCls(), is_v3=True) is not None  # type: ignore[arg-type]
    assert tf_api.convert_keras_to_onnx(b"not_a_zip", is_v3=True) is not None
    assert tf_api.convert_tflite_to_onnx(b"TFL3\x00\x00\x00\x00") is not None


def test_keras_h5_torch_script_fx_and_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover 100% of keras_h5_parser, torch/script, torch/fx, frontend/ast_parser, frontend/weight_utils, frontend/tracer, frontend/nn."""
    import json

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. KerasH5Parser with Functional model, Sequential model, unsupported class, missing model_config, and weights
        h5_func_path = Path(tmpdir) / "func.h5"
        func_cfg = {
            "class_name": "Model",
            "config": {
                "layers": [
                    {"name": "in1", "class_name": "InputLayer", "inbound_nodes": [], "config": {}},
                    {
                        "name": "d1",
                        "class_name": "Dense",
                        "inbound_nodes": [[["in1", 0, 0, {}]]],
                        "config": {"units": 4},
                    },
                ]
            },
        }
        with h5py.File(h5_func_path, "w") as f:
            f.attrs["model_config"] = json.dumps(func_cfg).encode("utf-8")
            mw = f.create_group("model_weights")
            lg = mw.create_group("d1")
            sub = lg.create_group("d1")
            sub.create_dataset("kernel:0", data=np.ones((2, 4), dtype=np.float32))

        p_h5 = keras_h5_parser.KerasH5Parser(filename=str(h5_func_path))
        assert len(p_h5.parse().nodes) == 2
        w_h5 = p_h5.get_weights()
        assert "d1" in w_h5 and len(w_h5["d1"]) == 1
        p_h5.close()

        # Sequential H5 via bytes
        h5_seq_bytes = io.BytesIO()
        seq_cfg = {
            "class_name": "Sequential",
            "config": {
                "layers": [
                    {"class_name": "Dense", "config": {"name": "s1", "units": 2}},
                    {"class_name": "Relu", "config": {"name": "s2"}},
                ]
            },
        }
        with h5py.File(h5_seq_bytes, "w") as f:
            f.attrs["model_config"] = json.dumps(seq_cfg)
        assert len(tf_api.convert_keras_to_onnx(h5_seq_bytes.getvalue()).nodes) >= 1

        # Unsupported class H5
        h5_unsup = io.BytesIO()
        with h5py.File(h5_unsup, "w") as f:
            f.attrs["model_config"] = json.dumps({"class_name": "Subclassed"})
        with pytest.raises(ConversionError):
            keras_h5_parser.KerasH5Parser(data=h5_unsup.getvalue()).parse()

        # Missing model_config H5
        h5_nomc = io.BytesIO()
        with h5py.File(h5_nomc, "w") as f:
            pass
        with pytest.raises(ConversionError):
            keras_h5_parser.KerasH5Parser(data=h5_nomc.getvalue()).parse()

        # Empty KerasH5Parser (f is None)
        p_empty = keras_h5_parser.KerasH5Parser()
        assert len(p_empty.parse().nodes) == 1
        assert p_empty.get_weights() == {}

        # h5py is None branch
        monkeypatch.setattr(keras_h5_parser, "h5py", None)
        with pytest.raises(ImportError):
            keras_h5_parser.KerasH5Parser()
        monkeypatch.undo()

    # 2. TorchScriptParser (torch/script.py)
    class MockTorchTensor:
        """Test helper class MockTorchTensor."""

        def __init__(self, arr: np.ndarray, dtype: str = "float32") -> None:
            """Test helper function __init__."""
            self._arr = arr
            self.shape = arr.shape
            self.dtype = dtype

        def numpy(self) -> np.ndarray:
            """Test helper function numpy."""
            return self._arr

    monkeypatch.setattr(torch_script.torch, "Tensor", MockTorchTensor)

    class MockValType:
        """Test helper class MockValType."""

        def __init__(
            self,
            kind_str: str,
            st: str | None = None,
            sizes: list[int] | None = None,
            raise_sizes: bool = False,
        ) -> None:
            """Test helper function __init__."""
            self._kind = kind_str
            self._st = st
            self._sizes = sizes
            self._raise = raise_sizes

        def kind(self) -> str:
            """Test helper function kind."""
            return self._kind

        def scalarType(self) -> str:
            """Test helper function scalarType."""
            return self._st or "float32"

        def sizes(self) -> list[int] | None:
            """Test helper function sizes."""
            if self._raise:
                raise RuntimeError("no sizes")
            return self._sizes

    class MockVal:
        """Test helper class MockVal."""

        def __init__(self, name: str, vtype: MockValType) -> None:
            """Test helper function __init__."""
            self._name = name
            self._vtype = vtype

        def debugName(self) -> str:
            """Test helper function debugName."""
            return self._name

        def type(self) -> MockValType:
            """Test helper function type."""
            return self._vtype

    class MockTSNode:
        """Test helper class MockTSNode."""

        def __init__(
            self,
            kind_str: str,
            ins: list[MockVal],
            outs: list[MockVal],
            attr_mode: str = "",
            blocks: list[object] | None = None,
        ) -> None:
            """Test helper function __init__."""
            self._kind = kind_str
            self._ins = ins
            self._outs = outs
            self._mode = attr_mode
            self._blocks = blocks or []

        def kind(self) -> str:
            """Test helper function kind."""
            return self._kind

        def inputs(self) -> list[MockVal]:
            """Test helper function inputs."""
            return self._ins

        def outputs(self) -> list[MockVal]:
            """Test helper function outputs."""
            return self._outs

        def attributeNames(self) -> list[str]:
            """Test helper function attributeNames."""
            return ["other", "value"] if self._mode else []

        def t(self, _k: str) -> MockTorchTensor:
            """Test helper function t."""
            if self._mode == "t":
                return MockTorchTensor(np.ones((2,), dtype=np.float32), "float32")
            raise RuntimeError("not t")

        def f(self, _k: str) -> float:
            """Test helper function f."""
            if self._mode == "f":
                return 1.5
            raise RuntimeError("not f")

        def i(self, _k: str) -> int:
            """Test helper function i."""
            if self._mode == "i":
                return 42
            raise RuntimeError("not i")

        def s(self, _k: str) -> str:
            """Test helper function s."""
            if self._mode == "s":
                return "str_val"
            raise RuntimeError("not s")

        def blocks(self) -> list[object]:
            """Test helper function blocks."""
            return self._blocks

    v_self = MockVal("self", MockValType("ClassType"))
    v_f32 = MockVal("x_f32", MockValType("TensorType", "float32", [1, 2]))
    v_f64 = MockVal("x_f64", MockValType("TensorType", "float64", None, raise_sizes=True))
    v_i64 = MockVal("x_i64", MockValType("TensorType", "int64"))
    v_i32 = MockVal("x_i32", MockValType("TensorType", "int32"))
    v_b = MockVal("x_b", MockValType("TensorType", "bool"))
    v_unk = MockVal("x_unk", MockValType("TensorType", "unk_dtype"))
    v_bool_k = MockVal("k_b", MockValType("BoolType"))
    v_int_k = MockVal("k_i", MockValType("IntType"))
    v_flt_k = MockVal("k_f", MockValType("FloatType"))
    v_other_k = MockVal("k_o", MockValType("OtherType"))

    v_c1 = MockVal("c1", MockValType("TensorType"))
    v_c2 = MockVal("c2", MockValType("FloatType"))
    v_c3 = MockVal("c3", MockValType("IntType"))
    v_c4 = MockVal("c4", MockValType("StringType"))
    v_c5 = MockVal("c5", MockValType("NoneType"))
    v_c6 = MockVal("c6", MockValType("NoneType"))
    v_out1 = MockVal("out1", MockValType("TensorType", "float32", [1, 2]))
    v_out2 = MockVal("out2", MockValType("TensorType", "float32", None, raise_sizes=True))

    from onnx9000.core.registry import register_op

    @register_op("torch", "custom_ts_reg")
    def _custom_ts_reg(
        inputs: list[object], outputs: list[object], params: dict[str, object]
    ) -> FENode:
        """Test helper function _custom_ts_reg."""
        return FENode("CustomTS", inputs=inputs, outputs=outputs, name="cust_ts")

    class MockTSGraph:
        """Test helper class MockTSGraph."""

        def inputs(self) -> list[MockVal]:
            """Test helper function inputs."""
            return [
                v_self,
                v_f32,
                v_f64,
                v_i64,
                v_i32,
                v_b,
                v_unk,
                v_bool_k,
                v_int_k,
                v_flt_k,
                v_other_k,
            ]

        def nodes(self) -> list[MockTSNode]:
            """Test helper function nodes."""
            return [
                MockTSNode("prim::Constant", [], [v_c1], "t"),
                MockTSNode("prim::Constant", [], [v_c2], "f"),
                MockTSNode("prim::Constant", [], [v_c3], "i"),
                MockTSNode("prim::Constant", [], [v_c4], "s"),
                MockTSNode("prim::Constant", [], [v_c5], "none"),
                MockTSNode("prim::Constant", [], [v_c6], ""),
                MockTSNode("aten::add", [v_f32, v_c1], [v_out1]),
                MockTSNode("aten::mul", [v_out1, v_c2], [v_out2]),
                MockTSNode("aten::sub", [v_out2, v_c3], [MockVal("o3", MockValType("TensorType"))]),
                MockTSNode("aten::div", [v_out2, v_c3], []),
                MockTSNode(
                    "aten::custom_ts_reg", [v_out1], [MockVal("o4", MockValType("TensorType"))]
                ),
                MockTSNode(
                    "prim::If", [v_b], [MockVal("o5", MockValType("TensorType"))], blocks=[1, 2]
                ),
                MockTSNode(
                    "prim::If", [v_b], [MockVal("o6", MockValType("TensorType"))], blocks=[1]
                ),
            ]

        def outputs(self) -> list[MockVal]:
            """Test helper function outputs."""
            return [v_out1, MockVal("unmapped_out", MockValType("TensorType"))]

    class MockScriptModule:
        """Test helper class MockScriptModule."""

        original_name = "MyScriptMod"
        graph = MockTSGraph()

    monkeypatch.setattr(torch_script.torch.jit, "ScriptModule", MockScriptModule)
    ts_gb = torch_script.TorchScriptParser(MockScriptModule()).parse()
    assert len(ts_gb.nodes) > 0

    class MockFuncObj:
        """Test helper class MockFuncObj."""

        __name__ = "my_ts_fn"
        graph = MockTSGraph()

    assert len(torch_script.TorchScriptParser(MockFuncObj()).parse().nodes) > 0

    # 3. FXParser (torch/fx.py)
    class MockFXNode:
        """Test helper class MockFXNode."""

        def __init__(
            self,
            op: str,
            name: str,
            target: object,
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
            meta: dict[str, object] | None = None,
        ) -> None:
            """Test helper function __init__."""
            self.op = op
            self.name = name
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.meta = meta or {}

    class MockFXTensor:
        """Test helper class MockFXTensor."""

        def __init__(self, arr: np.ndarray, dtype: str = "float32") -> None:
            """Test helper function __init__."""
            self._arr = arr
            self.shape = arr.shape
            self.dtype = dtype

        def detach(self) -> MockFXTensor:
            """Test helper function detach."""
            return self

        def cpu(self) -> MockFXTensor:
            """Test helper function cpu."""
            return self

        def numpy(self) -> np.ndarray:
            """Test helper function numpy."""
            return self._arr

    monkeypatch.setattr(torch_fx.torch, "Tensor", MockFXTensor)
    monkeypatch.setattr(torch_fx.torch.fx, "Node", MockFXNode)

    n_ph1 = MockFXNode(
        "placeholder",
        "x",
        "x",
        meta={"tensor_meta": types.SimpleNamespace(shape=(2, "s0"), dtype="float32")},
    )
    n_ph2 = MockFXNode("placeholder", "y", "y", meta={})
    n_attr1 = MockFXNode("get_attr", "w", "weight")
    n_attr2 = MockFXNode("get_attr", "non_t", "non_tensor")
    n_call_fn = MockFXNode(
        "call_function", "add_1", "custom_add_op", args=(n_ph1, 1.0), meta={"nn_module_stack": "s"}
    )
    n_call_mul = MockFXNode("call_method", "mul_1", "custom_mul_op", args=(n_call_fn, n_attr1))
    n_call_relu = MockFXNode("call_module", "relu_1", "sub_relu", args=(n_call_mul,))
    n_call_reg = MockFXNode("call_function", "reg_1", "custom_ts_reg", args=(n_call_relu,))
    n_out1 = MockFXNode("output", "out1", "output", args=(n_call_reg,))
    n_out2 = MockFXNode("output", "out2", "output", args=([n_call_reg, "not_a_node"],))
    n_out3 = MockFXNode("output", "out3", "output", args=(123,))
    n_other = MockFXNode("unknown_op", "unk", "unk")

    class MockSubRelu:
        """Test helper class MockSubRelu."""

        pass

    class MockGraphModule:
        """Test helper class MockGraphModule."""

        weight = MockFXTensor(np.ones((2, 2), dtype=np.float32))
        non_tensor = "not_a_tensor"
        graph = types.SimpleNamespace(
            nodes=[
                n_ph1,
                n_ph2,
                n_attr1,
                n_attr2,
                n_call_fn,
                n_call_mul,
                n_call_relu,
                n_call_reg,
                n_out1,
                n_out2,
                n_out3,
                n_other,
            ]
        )

        def named_buffers(self) -> list[tuple[str, object]]:
            """Test helper function named_buffers."""
            return [("weight", self.weight)]

        def get_submodule(self, _target: str) -> object:
            """Test helper function get_submodule."""
            return MockSubRelu()

    fx_p = torch_fx.FXParser(MockGraphModule())
    assert fx_p._get_shape(None) == ()
    assert len(fx_p.parse().nodes) > 0

    # 4. frontend/ast_parser.py, frontend/weight_utils.py, frontend/tracer.py
    def sample_script_fn(a: FETensor, b: FETensor) -> tuple[FETensor, list[FETensor]]:
        """Test helper function sample_script_fn."""
        if a:
            for _ in [a]:
                pass
        else:
            while False:
                pass
        return (a, [b, 123])  # type: ignore[list-item]

    sc = ast_parser.ScriptCompiler(sample_script_fn)
    t_fe = FETensor((2,), DType.FLOAT32, "a")
    sc.compile(t_fe)
    sc.generic_visit(ast.Pass())
    sc.visit_Return(ast.Return(value=ast.Name(id="a")))
    sc.visit_Return(ast.Return(value=None))
    sc.visit_If(ast.If(test=ast.Constant(value=1), body=[ast.Pass()], orelse=[]))

    # weight_utils.py
    g_wu = Graph("gwu")
    c_wu = Constant(
        "cwu",
        values=np.array([1.0, 2.0], dtype=np.float32).tobytes(),
        shape=(2,),
        dtype=DType.FLOAT32,
    )
    c_wu_none = Constant("cwu_n", values=None, shape=(2,), dtype=DType.FLOAT32)
    v_wu = Variable("vwu", shape=(2,), dtype=DType.FLOAT32)
    g_wu.add_tensor(c_wu)
    g_wu.add_tensor(c_wu_none)
    g_wu.add_tensor(v_wu)
    sd = fe_wutils.export_state_dict(g_wu)
    assert "cwu" in sd

    fake_torch_mod = types.ModuleType("torch")
    fake_torch_mod.from_numpy = lambda arr: arr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mod)
    sd_torch = fe_wutils.export_state_dict(g_wu)
    assert "cwu" in sd_torch
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert fe_wutils.universal_weight_bridge({"w": np.array([1.0])}, "pytorch") is not None
    assert fe_wutils.universal_weight_bridge({"w": np.array([1.0])}, "other") is not None

    # Test safetensors.torch import branch (both missing and mocked)
    assert fe_wutils.universal_weight_bridge({"w": np.array([1.0])}, "safetensors") is None
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save = lambda w: b"saved_st"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)
    assert fe_wutils.universal_weight_bridge({"w": np.array([1.0])}, "safetensors") == b"saved_st"


def test_remaining_python_converters_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover all remaining lines and branches across onnx9000-converters."""
    import importlib

    # 1. Caffe parser & mapper:
    # - parser.py line 37: "false" boolean value
    # - mapper.py 93->109, 101->109, 130->149, 141->149: Convolution & InnerProduct without weights, and with 1 weight blob
    c_proto = parse_prototxt(
        'name: "C"\nlayer {\n  name: "c_now"\n  type: "Convolution"\n  bottom: "x"\n  top: "h1"\n  convolution_param {\n    bias_term: false\n  }\n}\nlayer {\n  name: "c_1w"\n  type: "Convolution"\n  bottom: "h1"\n  top: "h2"\n}\nlayer {\n  name: "ip_now"\n  type: "InnerProduct"\n  bottom: "h2"\n  top: "h3"\n}\nlayer {\n  name: "ip_1w"\n  type: "InnerProduct"\n  bottom: "h3"\n  top: "y"\n}\n'
    )
    CaffeMapper(
        c_proto,
        {
            "c_1w": [np.ones((1, 1, 1, 1), dtype=np.float32)],
            "ip_1w": [np.ones((1, 1), dtype=np.float32)],
        },
    ).map()

    # 2. Darknet parser & mapper:
    parse_cfg("invalid_line\n")
    parse_cfg("")
    dm_blocks = parse_cfg(
        "[net]\nchannels=3\n[convolutional]\nfilters=2\nsize=3\nstride=2\npad=0\n[route]\nlayers=0\n[route]\nlayers=-1\n[shortcut]\nfrom=-1\nactivation=mish\n[shortcut]\nfrom=-1\nactivation=swish\n"
    )
    dm = DarknetMapper(dm_blocks, np.zeros(100, dtype=np.float32))
    assert dm.map() is not None
    dm_empty = DarknetMapper([{"type": "net", "channels": "3"}], np.zeros(1, dtype=np.float32))
    assert dm_empty.map() is not None

    # 3. frontend/bonsai.py, builder.py, exporter.py, jit.py, quantization.py, tracer.py, utils.py
    bonsai.BonsaiImporter(strip_zeros_to_sparse=False).import_model({})
    bonsai.BonsaiImporter().import_model(
        {
            "weights": {
                "w_non_arr": "not_an_ndarray",
                "w_dense": np.ones((4,), dtype=np.float32),
                "w_sparse": np.zeros((4,), dtype=np.float32),
            }
        }
    )

    gb = fe_builder.GraphBuilder("gb")
    t_gb = FETensor((1, 1, 2, 2), DType.FLOAT32, "t_gb")
    gb.inputs.append(t_gb)
    gb.inputs.append(t_gb)
    assert gb.to_graph() is not None
    gb.add_node(FENode("Relu", inputs=[t_gb, "unregistered_str"], outputs=[t_gb]))

    # exporter.py 32->31, 36->35: export with extra input_names/output_names beyond builder.inputs/outputs
    class SimpleMod(fe_module.Module):
        """Test helper class SimpleMod."""

        def forward(self, x: FETensor) -> FETensor:
            """Test helper function forward."""
            return x + x

    buf_exp = io.BytesIO()
    fe_exporter.export(
        SimpleMod(),
        (FETensor((2, 2), DType.FLOAT32, "x"),),
        buf_exp,
        input_names=["x", "extra_in"],
        output_names=["y", "extra_out"],
        dynamic_axes={"other_tensor": {0: "B"}, "x": {99: "oob"}},
    )

    # frontend/jit.py 37->36, 43->45: fn with non-Tensor arg and returning non-Tensor (None)
    @fe_jit.jit
    def jit_non_tensor(a: int) -> None:
        """Test helper function jit_non_tensor."""
        return None

    jit_non_tensor(42)

    # frontend/quantization.py 48->exit: GPTQParser when g_idx is None
    g_fq = Graph("gfq")
    fe_quant.GPTQParser().parse_state_dict(g_fq, {}, "w")

    # frontend/tracer.py 62-63, 84->86, 113->116, 128-132
    assert fe_tracer.Proxy("px").name == "px"
    traced_none = fe_tracer.trace(lambda a, _b: None, t_gb, t_gb)
    assert len(traced_none.outputs) == 0

    # script() both with torch in sys.modules (hitting lines 129-133) and without torch (falling back to ScriptCompiler)
    def script_target(x: FETensor) -> FETensor:
        """Test helper function script_target."""
        return x

    assert fe_tracer.script(script_target, t_gb) is not None
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setattr(
        torch_script,
        "TorchScriptParser",
        lambda _f: types.SimpleNamespace(parse=lambda: gb),
    )
    assert fe_tracer.script(script_target) is gb
    monkeypatch.undo()

    # frontend/utils.py 61->63, 132->143, 135->143: record_op with duplicate Parameter, 1-input Add, and 1-input MatMul
    p_dup = fe_module.Parameter((1,), DType.FLOAT32, "p_dup")
    with fe_tracer.Tracer(gb):
        fe_utils.record_op("Relu", [p_dup])
        fe_utils.record_op("Relu", [p_dup])
        fe_utils.record_op("Add", [t_gb])
        fe_utils.record_op("MatMul", [t_gb])
        fe_utils.record_op("CustomEmpty", [])

    # 4. frontend/nn/conv.py, functional.py, init.py, normalization.py, rnn.py
    assert fe_conv._pair((3, 3)) == (3, 3)
    assert fe_conv._single((3,)) == (3,)
    assert fe_conv._triple((3, 3, 3)) == (3, 3, 3)
    fe_conv.Conv1d(3, 4, 3, bias=True)
    fe_conv.Conv2d(3, 4, 3, bias=False)
    fe_conv.Conv3d(3, 4, 3, bias=False)
    fe_conv.ConvTranspose1d(3, 4, 3, bias=True)
    fe_conv.ConvTranspose1d(3, 4, 3, bias=False)
    fe_conv.ConvTranspose2d(3, 4, 3, bias=False)

    with fe_tracer.Tracer(gb):
        fe_F.pad(t_gb, (1, 1), mode="reflect")
        fe_F.interpolate(t_gb, size=(4, 4))
        fe_F.interpolate(t_gb, scale_factor=(2.0, 2.0))
        fe_F.interpolate(t_gb, scale_factor="invalid")

    t_1d = FETensor((4,), DType.FLOAT32, "t1d", np.zeros((4,), dtype=np.float32))
    t_4d = FETensor((4, 4, 3, 3), DType.FLOAT32, "t4d", np.zeros((4, 4, 3, 3), dtype=np.float32))
    t_no_data = FETensor((4, 4), DType.FLOAT32, "tnd", None)
    with pytest.raises(ConversionError):
        fe_init.calculate_fan_in_and_fan_out(t_1d)
    assert fe_init.calculate_fan_in_and_fan_out(t_4d) == (36, 36)
    fe_init.kaiming_uniform_(t_no_data)
    fe_init.kaiming_normal_(t_no_data)
    fe_init.xavier_uniform_(t_no_data)
    fe_init.xavier_normal_(t_no_data)
    fe_init.constant_(t_no_data, 1.0)

    bn_no_stats = fe_norm.BatchNorm2d(4, affine=False, track_running_stats=False)
    bn3d = fe_norm.BatchNorm3d(4)
    in_with_stats = fe_norm.InstanceNorm1d(4, affine=True, track_running_stats=True)
    in2_with_stats = fe_norm.InstanceNorm2d(4, affine=True, track_running_stats=True)
    in3_with_stats = fe_norm.InstanceNorm3d(4, affine=True, track_running_stats=True)
    ln_no_affine = fe_norm.LayerNorm(4, elementwise_affine=False)
    gn_no_affine = fe_norm.GroupNorm(2, 4, affine=False)
    with fe_tracer.Tracer(gb):
        bn_no_stats(t_gb)
        bn3d(t_gb)
        in_with_stats(t_gb)
        in2_with_stats(t_gb)
        in3_with_stats(t_gb)
        ln_no_affine(t_gb)
        gn_no_affine(t_gb)
        rnn_mod = fe_rnn.RNN(4, 4, bias=False)
        monkeypatch.setattr(fe_utils, "record_op", lambda *_a, **_kw: [t_gb])
        bn_no_stats(t_gb)
        monkeypatch.setattr(fe_utils, "record_op", lambda *_a, **_kw: t_gb)
        rnn_mod(t_gb)
        monkeypatch.undo()

    # 5. jit/__init__.py, compiler.py, hasher.py
    g_jit = Graph("gjit")
    t_j_arr = Tensor(
        "tj_arr", (1,), DType.FLOAT32, is_initializer=True, data=np.array([1.0], dtype=np.float32)
    )
    t_j_bytes = Tensor(
        "tj_bytes", (1,), DType.FLOAT32, is_initializer=True, data=b"\x00\x00\x80\x3f"
    )
    g_jit.add_tensor(t_j_arr)
    g_jit.add_tensor(t_j_bytes)
    assert jit.hasher.hash_graph(g_jit) is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        monkeypatch.setattr(jit, "load", lambda _p: g_jit)
        monkeypatch.setattr(jit, "plan_memory", lambda _g: None)
        monkeypatch.setattr(jit, "compile_wasm", lambda _g, d: d / "out.js")
        assert jit.compile("fake.onnx", target="wasm", out_dir=tmp_p) == tmp_p / "out.js"
        assert jit.compile("fake.onnx", target="other") is None
        monkeypatch.undo()

        # compile_wasm with ONNX9000_DEBUG=True and fallback to emcc when ONNX9000_WASM_COMPILER not found
        monkeypatch.setattr(jit.compiler.config, "ONNX9000_DEBUG", True)
        monkeypatch.setattr(jit.compiler.config, "ONNX9000_WASM_COMPILER", "nonexistent_emcc")
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/emcc" if cmd == "emcc" else None)
        monkeypatch.setattr("subprocess.run", lambda *_a, **_kw: None)
        js_out = jit.compiler.compile_wasm(g_jit, tmp_p / "out")
        js_out.write_text("// cached js")
        js_out.with_suffix(".wasm").write_bytes(b"wasm")
        jit.compiler.compile_wasm(g_jit, tmp_p / "out")

        # compile_cpp on darwin with ONNX9000_USE_ACCELERATE=False and ONNX9000_DEBUG=True
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr(jit.compiler.config, "ONNX9000_USE_ACCELERATE", False)
        monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/clang++")
        monkeypatch.setattr(jit.compiler, "PackageLoader", lambda *_a: None)
        monkeypatch.setattr(
            jit.compiler,
            "Environment",
            lambda **_kw: types.SimpleNamespace(
                get_template=lambda _n: types.SimpleNamespace(render=lambda **_k: "// tpl")
            ),
        )
        jit.compiler.compile_cpp(g_jit)
        monkeypatch.undo()

    # 6. mltools/libsvm.py, sparkml.py, mxnet/mapper.py, ncnn/parser.py, ncnn/weights.py, onnx_parser.py
    libsvm.parse_libsvm("svm_type c_svc\nnr_class 2\nSV\n1.0 1:0.5\n\n")
    sparkml.parse_sparkml_pipeline({"stages": [{"class": "Other", "uid": "u"}]})
    sparkml.parse_sparkml_pipeline('{"class": "Other"}')
    MXNetMapper(
        {
            "nodes": [
                {"op": "null", "name": "x"},
                {
                    "op": "Activation",
                    "name": "a",
                    "attrs": {"act_type": "tanh"},
                    "inputs": [[0, 0, 0]],
                },
            ],
            "heads": [[1, 0, 0], [1, 0, 0]],
        },
        {},
    ).map()
    parse_param("7767517\n1 1\nInput in 0 1 data 0=3 bad_token_without_equals\n")

    # ncnn/weights.py: num_elements=0, empty stream, FP16 magic 0x01306B47, INT8 magic 0x000D4B38, and raw float32
    raw_float_bin = struct.pack("<2f", 1.25, 2.5)
    wr_raw = WeightsReader(io.BytesIO(raw_float_bin))
    assert len(wr_raw.read_blob(0)) == 0
    assert len(wr_raw.read_blob(2)) == 2
    assert len(wr_raw.read_blob(2)) == 2  # EOF -> empty tag_data
    wr_fp16 = WeightsReader(
        io.BytesIO(struct.pack("<I", 0x01306B47) + np.array([1.0, 2.0], dtype=np.float16).tobytes())
    )
    assert len(wr_fp16.read_blob(2)) == 2
    wr_int8 = WeightsReader(io.BytesIO(struct.pack("<I", 0x000D4B38) + b"\x01\x02"))
    assert len(wr_int8.read_blob(2)) == 2

    # onnx_parser.py line 213: unpacked dims (field 1, wire 0) in TensorProto (tested twice so dims already exists on 2nd)
    with tempfile.TemporaryDirectory() as tmpdir:
        f_op = Path(tmpdir) / "d.onnx"
        f_op.write_bytes(b"\x08\x02\x08\x03")
        with onnx_parser.PureOnnxParser(str(f_op)) as pop:
            res_tp = pop._parse_message(pop.view, 0, len(pop.view), "TensorProto")
            assert res_tp["dims"] == [2, 3]

    # 7. paddle/builder.py, parsers.py, passes.py, tensor_ops.py
    assert paddle_parsers.PaddleGraph() is not None
    assert skl_builder is not None
    from onnx9000.converters.paddle import passes as paddle_passes
    from onnx9000.converters.paddle import tensor_ops as paddle_tensor_ops
    from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder

    p_builder = PaddleToONNXGraphBuilder("p")
    p_node = paddle_parsers.PaddleNode("n", "relu", {"ShapeTensor": ["st"]}, {}, {"scalar_attr": 5})
    assert p_builder.extract_list_attr(p_node, "scalar_attr") == [5]
    assert p_builder.extract_list_attr(p_node, "missing_attr", default=[1, 2]) == [1, 2]
    paddle_tensor_ops._map_fill_constant(p_builder, p_node)
    p_builder.make_node("Relu", ["x"], {}, "n1")
    p_builder.replace_node("nonexistent", Node("Identity", ["x"], ["y"], name="id"))

    # PaddleGraph.topological_sort with parent found vs not found, and load_paddle_model with feed/fetch vars not in main_block.vars
    pg = paddle_parsers.PaddleGraph()
    pb_blk = paddle_parsers.PaddleBlock(0, -1)
    op_parent = paddle_parsers.PaddleNode(
        "p1", "relu", {"X": ["unproduced_in"]}, {"Out": ["mid"]}, {}
    )
    op_child = paddle_parsers.PaddleNode("c1", "relu", {"X": ["mid"]}, {"Out": ["y"]}, {})
    op_feed = paddle_parsers.PaddleNode("f1", "feed", {}, {"Out": ["missing_feed_var"]}, {})
    op_fetch = paddle_parsers.PaddleNode("f2", "fetch", {"X": ["missing_fetch_var"]}, {}, {})
    pb_blk.ops.extend([op_child, op_parent, op_feed, op_fetch])
    pg.blocks.append(pb_blk)
    assert len(pg.topological_sort()) == 4
    monkeypatch.setattr(
        paddle_parsers,
        "PaddleProtobufParser",
        lambda _d: types.SimpleNamespace(parse_framework=lambda: pg),
    )
    paddle_parsers.load_paddle_model(b"dummy")
    monkeypatch.undo()

    # field 2 (len 6) -> s_field 2 (len 4) -> ss_field 1 (len 2) -> sss_field 2 (wire 0, val 5)
    p_parser = paddle_parsers.PaddleProtobufParser(b"\x12\x06\x12\x04\x0a\x02\x10\x05")
    assert p_parser.parse_var_desc(len(p_parser.data)).shape == [5]

    g_pp = Graph("gpp")
    g_pp.nodes.extend([Node("Identity", [], []), Node("Dropout", [], [])])
    paddle_passes.identity_removal_pass(g_pp)
    paddle_passes.dropout_removal_pass(g_pp)

    # 8. pytorch_fx_parser.py & safetensors/loader.py
    from onnx9000.converters import pytorch_fx_parser
    from onnx9000.converters.safetensors import loader as st_loader

    pytorch_fx_parser.load_pytorch_fx(
        '{"nodes": [{"op": "unknown"}, {"op": "output", "args": []}, {"op": "output", "args": [[123, "existing"]]}]}'
    )
    assert pytorch_fx_parser.PyTorchFXParser() is not None
    # Test when arg is already in graph.tensors (231->233)
    pytorch_fx_parser.load_pytorch_fx(
        '{"nodes": [{"op": "placeholder", "out": "x"}, {"op": "output", "args": [["x"]]}]}'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        from onnx9000.toolkit.safetensors.parser import save

        st_file = str(Path(tmpdir) / "test.safetensors")
        raw_hdr = b'{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
        Path(st_file).write_bytes(
            struct.pack("<Q", len(raw_hdr)) + raw_hdr + np.ones((2, 2), dtype=np.float32).tobytes()
        )
        g_st = Graph("gst")
        g_st.initializers.append("w")
        g_st.inputs.append(ValueInfo("w", (2, 2), DType.FLOAT32))
        g_st.nodes.append(Node("Relu", ["w"], ["out"]))
        g_st.nodes.append(
            Node(
                "Constant", [], ["c"], attributes={"value": Tensor("other_c", (1,), DType.FLOAT32)}
            )
        )
        st_loader.load_safetensors_to_graph(st_file, g_st)

        # validate_onnx_shapes_and_dtypes when name not in graph.tensors, or tensor has no shape/dtype
        g_val = Graph("gval")
        t_no_sh_dt = Tensor("w", (), None)
        g_val.tensors["w"] = t_no_sh_dt
        st_loader.validate_onnx_shapes_and_dtypes(st_file, g_val)
        st_loader.validate_onnx_shapes_and_dtypes(st_file, Graph("empty"))

        monkeypatch.setattr(
            "onnx9000.toolkit.safetensors.parser.save_file",
            lambda tensors, fn, metadata=None: Path(fn).write_bytes(save(tensors, metadata)),
            raising=False,
        )
        g_dump = Graph("dump_g")
        g_dump.tensors["t_none"] = Tensor("t_none", (1,), DType.FLOAT32, data=None)
        st_loader.dump_graph_to_safetensors(g_dump, st_file, topology_filename=None)
        monkeypatch.undo()

    st_loader.map_huggingface_to_onnx(
        {
            "c_attn.weight": np.ones((2, 4), dtype=np.float32),
            "c_attn.bias": np.ones((4,), dtype=np.float32),
        }
    )

    # 9. sklearn/builder.py, linear.py, preprocessing.py, svm.py, trees.py
    from onnx9000.converters.sklearn import (
        linear as skl_lin,
    )
    from onnx9000.converters.sklearn import (
        preprocessing as skl_prep,
    )
    from onnx9000.converters.sklearn import (
        svm as skl_svm,
    )
    from onnx9000.converters.sklearn import (
        trees as skl_trees,
    )

    g_skl = Graph("gskl")
    b_skl = skl_builder.SKLearnParser(
        types.SimpleNamespace(),
        "skl",
        initial_types=[("X", DType.FLOAT32, (None, 4))],
    )
    FeatureUnionCls = type("FeatureUnion", (), {"transformer_list": [("drop_step", "drop")]})
    StandardScalerCls = type("StandardScaler", (), {})
    b_skl._parse_estimator(FeatureUnionCls(), ["X"])
    b_skl._parse_estimator(StandardScalerCls(), ["X"])

    # decision tree classifier without classes_
    fake_tree = types.SimpleNamespace(
        children_left=np.array([-1]),
        children_right=np.array([-1]),
        feature=np.array([-2]),
        threshold=np.array([0.0]),
        value=np.array([[[1.0, 2.0]]]),
    )
    skl_trees.convert_decision_tree_classifier(
        types.SimpleNamespace(tree_=fake_tree),
        ["X"],
        g_skl,
    )

    # linearregressor / classifier without coef_ / intercept_ / classes_
    skl_lin._convert_linear_regressor(types.SimpleNamespace(), ["X"], g_skl)
    skl_lin._convert_linear_classifier(types.SimpleNamespace(), ["X"], g_skl)

    # preprocessing scalers without mean_/scale_/min_/center_ and normalizer with unknown norm
    skl_prep.convert_standard_scaler(types.SimpleNamespace(), ["X"], g_skl)
    skl_prep.convert_min_max_scaler(types.SimpleNamespace(), ["X"], g_skl)
    skl_prep.convert_max_abs_scaler(types.SimpleNamespace(), ["X"], g_skl)
    skl_prep.convert_robust_scaler(types.SimpleNamespace(), ["X"], g_skl)
    skl_prep.convert_normalizer(types.SimpleNamespace(norm="unknown"), ["X"], g_skl)

    # svm classifier / regressor without optional attributes
    skl_svm._convert_svm_classifier(types.SimpleNamespace(), ["X"], g_skl)
    skl_svm._convert_svm_regressor(types.SimpleNamespace(), ["X"], g_skl)

    # 10. tf/nn_ops.py, parsers.py, passes.py
    tf_b = tf_builder.TFToONNXGraphBuilder("tfb")
    tf_nn._map_conv2d(
        tf_b,
        tf_parsers.TFNode(
            "c_nhwc_2",
            "Conv2D",
            ["x", "w"],
            {"data_format": "NHWC", "strides": [1, 1], "dilations": [1, 1]},
        ),
    )
    tf_nn._map_conv2d(
        tf_b,
        tf_parsers.TFNode(
            "c_nchw_2",
            "Conv2D",
            ["x", "w"],
            {"data_format": "NCHW", "strides": [1, 1], "dilations": [1, 1]},
        ),
    )

    # tf/parsers.py 120->118, 123->118: extract_subgraph with duplicate visited node and missing node
    tfg_sub = tf_parsers.TFGraph(
        [
            tf_parsers.TFNode("n1", "Identity", ["missing_in"]),
            tf_parsers.TFNode("n2", "Add", ["n1", "n1"]),
        ]
    )
    tfg_sub.extract_subgraph(["n2", "n1"])

    # tf/parsers.py 351->360: load_keras_v3 with zip file not containing config.json
    with tempfile.TemporaryDirectory() as tmpdir:
        z_no_cfg = Path(tmpdir) / "nocfg.zip"
        with zipfile.ZipFile(z_no_cfg, "w") as zf:
            zf.writestr("other.txt", "hello")
        monkeypatch.setitem(sys.modules, "keras", types.SimpleNamespace(Model=type("M", (), {})))
        tf_parsers.load_keras_v3(z_no_cfg.read_bytes())
        monkeypatch.undo()

    # tf/passes.py 27->23, 48->44, 70->66, 90->98, 93->98:
    # Identity, Dropout, StopGradient with empty inputs/outputs, Transpose whose parent is not Transpose, and Transpose without perm
    g_tfp = Graph("gtfp")
    g_tfp.nodes.extend(
        [
            Node("Identity", [], []),
            Node("Dropout", [], []),
            Node("Custom_TFStopGradient", [], []),
            Node("Relu", ["x"], ["r_parent"]),
            Node("Transpose", ["r_parent"], ["t_after_relu"], attributes={"perm": [1, 0]}),
            Node("Transpose", ["x"], ["t_no_perm1"]),
            Node("Transpose", ["t_no_perm1"], ["t_no_perm2"]),
        ]
    )
    tf_passes.identity_removal_pass(g_tfp)
    tf_passes.dropout_removal_pass(g_tfp)
    tf_passes.remove_debug_nodes_pass(g_tfp)
    tf_passes.transpose_optimizer_pass(g_tfp)

    # 11. Reload keras_h5_parser, torch/fx, torch/script with missing optional dependencies and with mock torch present
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch.fx", types.SimpleNamespace())
    importlib.reload(torch_fx)
    monkeypatch.undo()

    orig_imp = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def block_optional_imports(name: str, *args: object, **kwargs: object) -> object:
        """Test helper function block_optional_imports."""
        if name in ("h5py", "torch", "torch.fx"):
            raise ImportError(f"blocked {name}")
        return orig_imp(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", block_optional_imports)
    importlib.reload(keras_h5_parser)
    importlib.reload(torch_fx)
    importlib.reload(torch_script)
    monkeypatch.undo()
    importlib.reload(keras_h5_parser)
    importlib.reload(torch_fx)
    importlib.reload(torch_script)

    # torch/fx.py line 159: call_function with op containing "relu"
    class MockFXNodeRelu:
        """Test helper class MockFXNodeRelu."""

        op = "call_function"
        name = "my_relu"
        target = "my_relu_fn"
        args = ()
        kwargs: dict[str, object] = {}
        meta: dict[str, object] = {}

    class MockGMRelu:
        """Test helper class MockGMRelu."""

        graph = types.SimpleNamespace(nodes=[MockFXNodeRelu()])

        def named_buffers(self) -> list[tuple[str, object]]:
            """Test helper function named_buffers."""
            return []

    torch_fx.FXParser(MockGMRelu()).parse()

    # torch/script.py lines 313, 315: aten::add and aten::mul fallback when not in registry
    from onnx9000.core.exceptions import UnsupportedOpError
    from onnx9000.core.registry import global_registry

    def _raise_unsup(domain: str, op_name: str, provider: str | None = None) -> object:
        """Test helper function _raise_unsup."""
        raise UnsupportedOpError(op_name)

    monkeypatch.setattr(global_registry, "get_op", _raise_unsup)

    class MockValSimple:
        """Test helper class MockValSimple."""

        def debugName(self) -> str:
            """Test helper function debugName."""
            return "v"

        def type(self) -> object:
            """Test helper function type."""
            return types.SimpleNamespace(kind=lambda: "FloatType")

    class MockNodeSubDiv:
        """Test helper class MockNodeSubDiv."""

        def __init__(self, k: str) -> None:
            """Test helper function __init__."""
            self._k = k

        def kind(self) -> str:
            """Test helper function kind."""
            return self._k

        def inputs(self) -> list[object]:
            """Test helper function inputs."""
            return []

        def outputs(self) -> list[object]:
            """Test helper function outputs."""
            return [MockValSimple()]

    class MockGraphSubDiv:
        """Test helper class MockGraphSubDiv."""

        def inputs(self) -> list[object]:
            """Test helper function inputs."""
            return []

        def nodes(self) -> list[object]:
            """Test helper function nodes."""
            return [
                MockNodeSubDiv("aten::add"),
                MockNodeSubDiv("aten::mul"),
                MockNodeSubDiv("aten::sub"),
                MockNodeSubDiv("aten::div"),
            ]

        def outputs(self) -> list[object]:
            """Test helper function outputs."""
            return []

    torch_script.TorchScriptParser(
        types.SimpleNamespace(__name__="sd", graph=MockGraphSubDiv())
    ).parse()
    monkeypatch.undo()
