"""Comprehensive unit tests for all Keras layer to ONNX converter functions."""

from __future__ import annotations

from onnx9000.converters.tf import keras_layers
from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def test_keras_conv_pool_rnn_bases() -> None:
    """Verify _map_keras_conv_base, _map_keras_pool_base, _map_keras_rnn_base, and _map_keras_conv_lstm_base."""
    builder = TFToONNXGraphBuilder("test_bases")

    out_empty_conv = keras_layers._map_keras_conv_base(
        builder, TFNode("c_empty", "Conv1D", inputs=[])
    )
    assert len(out_empty_conv) == 1
    assert builder.graph.nodes[-1].op_type == "Constant"

    node_conv = TFNode(
        "c_causal",
        "Conv1D",
        inputs=["x", "w"],
        attr={"dilation_rate": [2], "groups": 1, "padding": "causal", "kernel_size": [3]},
    )
    out_conv = keras_layers._map_keras_conv_base(builder, node_conv)
    assert len(out_conv) == 1
    assert builder.graph.nodes[-1].op_type == "Conv"
    assert builder.graph.nodes[-1].attributes["auto_pad"] == b"VALID"

    node_valid = TFNode("c_valid", "Conv2D", inputs=["x", "w"], attr={"padding": "valid"})
    keras_layers._map_keras_conv_base(builder, node_valid)
    assert builder.graph.nodes[-1].attributes["auto_pad"] == b"VALID"

    node_same = TFNode("c_same", "Conv2D", inputs=["x", "w"], attr={"padding": "same"})
    keras_layers._map_keras_conv_base(builder, node_same)
    assert builder.graph.nodes[-1].attributes["auto_pad"] == b"SAME_UPPER"

    out_empty_pool = keras_layers._map_keras_pool_base(
        builder, TFNode("p_empty", "MaxPool", inputs=[]), "MaxPool"
    )
    assert len(out_empty_pool) == 1
    assert builder.graph.nodes[-1].op_type == "Constant"

    for pad_mode, expected in [("valid", b"VALID"), ("same", b"SAME_UPPER"), ("custom", b"CUSTOM")]:
        node_p = TFNode(f"p_{pad_mode}", "MaxPool", inputs=["x"], attr={"padding": pad_mode})
        keras_layers._map_keras_pool_base(builder, node_p, "MaxPool")
        assert builder.graph.nodes[-1].op_type == "MaxPool"
        assert builder.graph.nodes[-1].attributes["auto_pad"] == expected

    out_empty_rnn = keras_layers._map_keras_rnn_base(
        builder, TFNode("r_empty", "LSTM", inputs=[]), "LSTM"
    )
    assert len(out_empty_rnn) == 1
    assert builder.graph.nodes[-1].op_type == "Constant"

    node_lstm_state = TFNode(
        "r_lstm",
        "LSTM",
        inputs=["x", "w", "r"],
        attr={
            "return_sequences": True,
            "return_state": True,
            "go_backwards": True,
            "time_major": True,
        },
    )
    out_lstm = keras_layers._map_keras_rnn_base(builder, node_lstm_state, "LSTM")
    assert len(out_lstm) == 3

    for dim in [1, 2, 3]:
        out_clstm = keras_layers._map_keras_conv_lstm_base(
            builder, TFNode(f"clstm_{dim}", "ConvLSTM", inputs=["x"]), dim
        )
        assert len(out_clstm) == 2
        assert builder.graph.nodes[-1].op_type == "Identity"
        out_clstm_empty = keras_layers._map_keras_conv_lstm_base(
            builder, TFNode(f"clstm_e_{dim}", "ConvLSTM", inputs=[]), dim
        )
        assert len(out_clstm_empty) == 1
        assert builder.graph.nodes[-1].op_type == "Constant"

    for layer_attr in [
        {"layer": {"class_name": "GRU"}},
        {"layer": {"class_name": "Other"}},
        {"layer": "not_a_dict"},
    ]:
        out_bi = keras_layers._map_keras_bidirectional(
            builder, TFNode("bi_test", "Bi", inputs=["x"], attr=layer_attr)
        )
        assert len(out_bi) == 1

    out_stft = keras_layers._map_keras_stft(
        builder,
        TFNode(
            "stft_custom",
            "STFT",
            inputs=["x"],
            attr={"sequence_length": 1024, "sequence_stride": 256},
        ),
    )
    assert len(out_stft) == 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    out_einsum2 = keras_layers._map_keras_einsum_dense(
        builder,
        TFNode("einsum2", "EinsumDense", inputs=["x", "w"], attr={"equation": b"ij,jk->ik"}),
    )
    assert len(out_einsum2) == 1
    assert builder.graph.nodes[-1].op_type == "Einsum"

    for fn_val, expected_op in [(b"tf.sqrt(x)", "Sqrt"), ("backend.exp(x)", "Exp")]:
        keras_layers._map_keras_layers__lambda(
            builder,
            TFNode("lam", "Lambda", inputs=["x"], attr={"function": fn_val}),
        )
        assert builder.graph.nodes[-1].op_type == expected_op


def test_keras_cropping_upsampling_zero_padding_variants() -> None:
    """Verify 1D/2D/3D cropping, upsampling, and zero_padding branches."""
    builder = TFToONNXGraphBuilder("test_variants")
    for dim in ["1D", "2D", "3D"]:
        crop_fn = keras_layers._map_keras_cropping(dim)
        crop_fn(builder, TFNode(f"crop_empty_{dim}", "Crop", inputs=[]))
        assert builder.graph.nodes[-1].op_type == "Constant"
        crop_fn(builder, TFNode(f"crop_int_{dim}", "Crop", inputs=["x"], attr={"cropping": 2}))
        assert builder.graph.nodes[-1].op_type == "Slice"

        up_fn = keras_layers._map_keras_upsampling(dim)
        up_fn(builder, TFNode(f"up_empty_{dim}", "Up", inputs=[]))
        assert builder.graph.nodes[-1].op_type == "Constant"
        up_fn(
            builder,
            TFNode(
                f"up_int_{dim}", "Up", inputs=["x"], attr={"size": 2, "interpolation": b"bilinear"}
            ),
        )
        assert builder.graph.nodes[-1].op_type == "Resize"

        zp_fn = keras_layers._map_keras_zero_padding(dim)
        zp_fn(builder, TFNode(f"zp_empty_{dim}", "ZP", inputs=[]))
        assert builder.graph.nodes[-1].op_type == "Constant"
        zp_fn(builder, TFNode(f"zp_int_{dim}", "ZP", inputs=["x"], attr={"padding": 2}))
        assert builder.graph.nodes[-1].op_type == "Pad"

    keras_layers._map_keras_cropping("2D")(
        builder, TFNode("c2d_tuple", "Crop", inputs=["x"], attr={"cropping": (1, 2)})
    )
    assert builder.graph.nodes[-1].op_type == "Slice"
    keras_layers._map_keras_cropping("3D")(
        builder, TFNode("c3d_tuple", "Crop", inputs=["x"], attr={"cropping": (1, 2, 3)})
    )
    assert builder.graph.nodes[-1].op_type == "Slice"
    keras_layers._map_keras_zero_padding("2D")(
        builder, TFNode("zp2d_tuple", "ZP", inputs=["x"], attr={"padding": (1, 2)})
    )
    assert builder.graph.nodes[-1].op_type == "Pad"
    keras_layers._map_keras_zero_padding("3D")(
        builder, TFNode("zp3d_tuple", "ZP", inputs=["x"], attr={"padding": (1, 2, 3)})
    )
    assert builder.graph.nodes[-1].op_type == "Pad"


def test__map_keras_conv1d() -> None:
    """Verify _map_keras_conv1d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_conv1d", "_map_keras_conv1d", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_conv1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode("empty__map_keras_conv1d", "_map_keras_conv1d", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_conv1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv1d_transpose() -> None:
    """Verify _map_keras_conv1d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv1d_transpose",
        "_map_keras_conv1d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_conv1d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_conv1d_transpose", "_map_keras_conv1d_transpose", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv1d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv2d() -> None:
    """Verify _map_keras_conv2d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_conv2d", "_map_keras_conv2d", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_conv2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode("empty__map_keras_conv2d", "_map_keras_conv2d", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_conv2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv2d_transpose() -> None:
    """Verify _map_keras_conv2d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv2d_transpose",
        "_map_keras_conv2d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_conv2d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_conv2d_transpose", "_map_keras_conv2d_transpose", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv2d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv3d() -> None:
    """Verify _map_keras_conv3d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_conv3d", "_map_keras_conv3d", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_conv3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode("empty__map_keras_conv3d", "_map_keras_conv3d", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_conv3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv3d_transpose() -> None:
    """Verify _map_keras_conv3d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv3d_transpose",
        "_map_keras_conv3d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_conv3d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_conv3d_transpose", "_map_keras_conv3d_transpose", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv3d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_depthwise_conv1d() -> None:
    """Verify _map_keras_depthwise_conv1d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_depthwise_conv1d",
        "_map_keras_depthwise_conv1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_depthwise_conv1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_depthwise_conv1d", "_map_keras_depthwise_conv1d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_depthwise_conv1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_depthwise_conv2d() -> None:
    """Verify _map_keras_depthwise_conv2d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_depthwise_conv2d",
        "_map_keras_depthwise_conv2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_depthwise_conv2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_depthwise_conv2d", "_map_keras_depthwise_conv2d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_depthwise_conv2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv1_d() -> None:
    """Verify _map_keras_layers__conv1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv1_d",
        "_map_keras_layers__conv1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv1_d", "_map_keras_layers__conv1_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__conv1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv1_d_transpose() -> None:
    """Verify _map_keras_layers__conv1_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv1_d_transpose",
        "_map_keras_layers__conv1_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv1_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__conv1_d_transpose",
        "_map_keras_layers__conv1_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv1_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv2_d() -> None:
    """Verify _map_keras_layers__conv2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv2_d",
        "_map_keras_layers__conv2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv2_d", "_map_keras_layers__conv2_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__conv2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv2_d_transpose() -> None:
    """Verify _map_keras_layers__conv2_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv2_d_transpose",
        "_map_keras_layers__conv2_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv2_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__conv2_d_transpose",
        "_map_keras_layers__conv2_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv2_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv3_d() -> None:
    """Verify _map_keras_layers__conv3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv3_d",
        "_map_keras_layers__conv3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv3_d", "_map_keras_layers__conv3_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__conv3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv3_d_transpose() -> None:
    """Verify _map_keras_layers__conv3_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv3_d_transpose",
        "_map_keras_layers__conv3_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv3_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__conv3_d_transpose",
        "_map_keras_layers__conv3_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv3_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution1_d() -> None:
    """Verify _map_keras_layers__convolution1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution1_d",
        "_map_keras_layers__convolution1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution1_d",
        "_map_keras_layers__convolution1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution1_d_transpose() -> None:
    """Verify _map_keras_layers__convolution1_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution1_d_transpose",
        "_map_keras_layers__convolution1_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution1_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution1_d_transpose",
        "_map_keras_layers__convolution1_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution1_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution2_d() -> None:
    """Verify _map_keras_layers__convolution2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution2_d",
        "_map_keras_layers__convolution2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution2_d",
        "_map_keras_layers__convolution2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution2_d_transpose() -> None:
    """Verify _map_keras_layers__convolution2_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution2_d_transpose",
        "_map_keras_layers__convolution2_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution2_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution2_d_transpose",
        "_map_keras_layers__convolution2_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution2_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution3_d() -> None:
    """Verify _map_keras_layers__convolution3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution3_d",
        "_map_keras_layers__convolution3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution3_d",
        "_map_keras_layers__convolution3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__convolution3_d_transpose() -> None:
    """Verify _map_keras_layers__convolution3_d_transpose emits ConvTranspose with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__convolution3_d_transpose",
        "_map_keras_layers__convolution3_d_transpose",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__convolution3_d_transpose(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode(
        "empty__map_keras_layers__convolution3_d_transpose",
        "_map_keras_layers__convolution3_d_transpose",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__convolution3_d_transpose(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__depthwise_conv1_d() -> None:
    """Verify _map_keras_layers__depthwise_conv1_d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__depthwise_conv1_d",
        "_map_keras_layers__depthwise_conv1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__depthwise_conv1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_layers__depthwise_conv1_d",
        "_map_keras_layers__depthwise_conv1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__depthwise_conv1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__depthwise_conv2_d() -> None:
    """Verify _map_keras_layers__depthwise_conv2_d emits Conv with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__depthwise_conv2_d",
        "_map_keras_layers__depthwise_conv2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__depthwise_conv2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_layers__depthwise_conv2_d",
        "_map_keras_layers__depthwise_conv2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__depthwise_conv2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__separable_conv1_d() -> None:
    """Verify _map_keras_layers__separable_conv1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__separable_conv1_d",
        "_map_keras_layers__separable_conv1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__separable_conv1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__separable_conv1_d",
        "_map_keras_layers__separable_conv1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__separable_conv1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__separable_conv2_d() -> None:
    """Verify _map_keras_layers__separable_conv2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__separable_conv2_d",
        "_map_keras_layers__separable_conv2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__separable_conv2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__separable_conv2_d",
        "_map_keras_layers__separable_conv2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__separable_conv2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__separable_convolution1_d() -> None:
    """Verify _map_keras_layers__separable_convolution1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__separable_convolution1_d",
        "_map_keras_layers__separable_convolution1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__separable_convolution1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__separable_convolution1_d",
        "_map_keras_layers__separable_convolution1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__separable_convolution1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__separable_convolution2_d() -> None:
    """Verify _map_keras_layers__separable_convolution2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__separable_convolution2_d",
        "_map_keras_layers__separable_convolution2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__separable_convolution2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__separable_convolution2_d",
        "_map_keras_layers__separable_convolution2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__separable_convolution2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_separable_conv1d() -> None:
    """Verify _map_keras_separable_conv1d emits Conv with valid inputs and Conv when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_separable_conv1d",
        "_map_keras_separable_conv1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_separable_conv1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_separable_conv1d", "_map_keras_separable_conv1d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_separable_conv1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"


def test__map_keras_separable_conv2d() -> None:
    """Verify _map_keras_separable_conv2d emits Conv with valid inputs and Conv when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_separable_conv2d",
        "_map_keras_separable_conv2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_separable_conv2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode(
        "empty__map_keras_separable_conv2d", "_map_keras_separable_conv2d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_separable_conv2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Conv"


def test__map_keras_avg_pooling_1d() -> None:
    """Verify _map_keras_avg_pooling_1d emits AveragePool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_avg_pooling_1d",
        "_map_keras_avg_pooling_1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_avg_pooling_1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "AveragePool"

    node_empty = TFNode(
        "empty__map_keras_avg_pooling_1d", "_map_keras_avg_pooling_1d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_avg_pooling_1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_avg_pooling_2d() -> None:
    """Verify _map_keras_avg_pooling_2d emits AveragePool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_avg_pooling_2d",
        "_map_keras_avg_pooling_2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_avg_pooling_2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "AveragePool"

    node_empty = TFNode(
        "empty__map_keras_avg_pooling_2d", "_map_keras_avg_pooling_2d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_avg_pooling_2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_avg_pooling_3d() -> None:
    """Verify _map_keras_avg_pooling_3d emits AveragePool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_avg_pooling_3d",
        "_map_keras_avg_pooling_3d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_avg_pooling_3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "AveragePool"

    node_empty = TFNode(
        "empty__map_keras_avg_pooling_3d", "_map_keras_avg_pooling_3d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_avg_pooling_3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_global_avg_pooling_1d() -> None:
    """Verify _map_keras_global_avg_pooling_1d emits GlobalAveragePool with valid inputs and GlobalAveragePool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_avg_pooling_1d",
        "_map_keras_global_avg_pooling_1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_avg_pooling_1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"

    node_empty = TFNode(
        "empty__map_keras_global_avg_pooling_1d",
        "_map_keras_global_avg_pooling_1d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_avg_pooling_1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"


def test__map_keras_global_avg_pooling_2d() -> None:
    """Verify _map_keras_global_avg_pooling_2d emits GlobalAveragePool with valid inputs and GlobalAveragePool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_avg_pooling_2d",
        "_map_keras_global_avg_pooling_2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_avg_pooling_2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"

    node_empty = TFNode(
        "empty__map_keras_global_avg_pooling_2d",
        "_map_keras_global_avg_pooling_2d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_avg_pooling_2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"


def test__map_keras_global_avg_pooling_3d() -> None:
    """Verify _map_keras_global_avg_pooling_3d emits GlobalAveragePool with valid inputs and GlobalAveragePool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_avg_pooling_3d",
        "_map_keras_global_avg_pooling_3d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_avg_pooling_3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"

    node_empty = TFNode(
        "empty__map_keras_global_avg_pooling_3d",
        "_map_keras_global_avg_pooling_3d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_avg_pooling_3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"


def test__map_keras_global_max_pooling_1d() -> None:
    """Verify _map_keras_global_max_pooling_1d emits GlobalMaxPool with valid inputs and GlobalMaxPool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_max_pooling_1d",
        "_map_keras_global_max_pooling_1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_max_pooling_1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"

    node_empty = TFNode(
        "empty__map_keras_global_max_pooling_1d",
        "_map_keras_global_max_pooling_1d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_max_pooling_1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"


def test__map_keras_global_max_pooling_2d() -> None:
    """Verify _map_keras_global_max_pooling_2d emits GlobalMaxPool with valid inputs and GlobalMaxPool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_max_pooling_2d",
        "_map_keras_global_max_pooling_2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_max_pooling_2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"

    node_empty = TFNode(
        "empty__map_keras_global_max_pooling_2d",
        "_map_keras_global_max_pooling_2d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_max_pooling_2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"


def test__map_keras_global_max_pooling_3d() -> None:
    """Verify _map_keras_global_max_pooling_3d emits GlobalMaxPool with valid inputs and GlobalMaxPool when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_global_max_pooling_3d",
        "_map_keras_global_max_pooling_3d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_global_max_pooling_3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"

    node_empty = TFNode(
        "empty__map_keras_global_max_pooling_3d",
        "_map_keras_global_max_pooling_3d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_global_max_pooling_3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"


def test__map_keras_layers__adaptive_average_pooling1_d() -> None:
    """Verify _map_keras_layers__adaptive_average_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_average_pooling1_d",
        "_map_keras_layers__adaptive_average_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_average_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_average_pooling1_d",
        "_map_keras_layers__adaptive_average_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_average_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__adaptive_average_pooling2_d() -> None:
    """Verify _map_keras_layers__adaptive_average_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_average_pooling2_d",
        "_map_keras_layers__adaptive_average_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_average_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_average_pooling2_d",
        "_map_keras_layers__adaptive_average_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_average_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__adaptive_average_pooling3_d() -> None:
    """Verify _map_keras_layers__adaptive_average_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_average_pooling3_d",
        "_map_keras_layers__adaptive_average_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_average_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_average_pooling3_d",
        "_map_keras_layers__adaptive_average_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_average_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__adaptive_max_pooling1_d() -> None:
    """Verify _map_keras_layers__adaptive_max_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_max_pooling1_d",
        "_map_keras_layers__adaptive_max_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_max_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_max_pooling1_d",
        "_map_keras_layers__adaptive_max_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_max_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__adaptive_max_pooling2_d() -> None:
    """Verify _map_keras_layers__adaptive_max_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_max_pooling2_d",
        "_map_keras_layers__adaptive_max_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_max_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_max_pooling2_d",
        "_map_keras_layers__adaptive_max_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_max_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__adaptive_max_pooling3_d() -> None:
    """Verify _map_keras_layers__adaptive_max_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__adaptive_max_pooling3_d",
        "_map_keras_layers__adaptive_max_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__adaptive_max_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__adaptive_max_pooling3_d",
        "_map_keras_layers__adaptive_max_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__adaptive_max_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__average_pooling1_d() -> None:
    """Verify _map_keras_layers__average_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__average_pooling1_d",
        "_map_keras_layers__average_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__average_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__average_pooling1_d",
        "_map_keras_layers__average_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__average_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__average_pooling2_d() -> None:
    """Verify _map_keras_layers__average_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__average_pooling2_d",
        "_map_keras_layers__average_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__average_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__average_pooling2_d",
        "_map_keras_layers__average_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__average_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__average_pooling3_d() -> None:
    """Verify _map_keras_layers__average_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__average_pooling3_d",
        "_map_keras_layers__average_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__average_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__average_pooling3_d",
        "_map_keras_layers__average_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__average_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__avg_pool1_d() -> None:
    """Verify _map_keras_layers__avg_pool1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__avg_pool1_d",
        "_map_keras_layers__avg_pool1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__avg_pool1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__avg_pool1_d", "_map_keras_layers__avg_pool1_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__avg_pool1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__avg_pool2_d() -> None:
    """Verify _map_keras_layers__avg_pool2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__avg_pool2_d",
        "_map_keras_layers__avg_pool2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__avg_pool2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__avg_pool2_d", "_map_keras_layers__avg_pool2_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__avg_pool2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__avg_pool3_d() -> None:
    """Verify _map_keras_layers__avg_pool3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__avg_pool3_d",
        "_map_keras_layers__avg_pool3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__avg_pool3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__avg_pool3_d", "_map_keras_layers__avg_pool3_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__avg_pool3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_average_pooling1_d() -> None:
    """Verify _map_keras_layers__global_average_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_average_pooling1_d",
        "_map_keras_layers__global_average_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_average_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_average_pooling1_d",
        "_map_keras_layers__global_average_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_average_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_average_pooling2_d() -> None:
    """Verify _map_keras_layers__global_average_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_average_pooling2_d",
        "_map_keras_layers__global_average_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_average_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_average_pooling2_d",
        "_map_keras_layers__global_average_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_average_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_average_pooling3_d() -> None:
    """Verify _map_keras_layers__global_average_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_average_pooling3_d",
        "_map_keras_layers__global_average_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_average_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_average_pooling3_d",
        "_map_keras_layers__global_average_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_average_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_avg_pool1_d() -> None:
    """Verify _map_keras_layers__global_avg_pool1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_avg_pool1_d",
        "_map_keras_layers__global_avg_pool1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_avg_pool1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_avg_pool1_d",
        "_map_keras_layers__global_avg_pool1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_avg_pool1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_avg_pool2_d() -> None:
    """Verify _map_keras_layers__global_avg_pool2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_avg_pool2_d",
        "_map_keras_layers__global_avg_pool2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_avg_pool2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_avg_pool2_d",
        "_map_keras_layers__global_avg_pool2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_avg_pool2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_avg_pool3_d() -> None:
    """Verify _map_keras_layers__global_avg_pool3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_avg_pool3_d",
        "_map_keras_layers__global_avg_pool3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_avg_pool3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_avg_pool3_d",
        "_map_keras_layers__global_avg_pool3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_avg_pool3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pool1_d() -> None:
    """Verify _map_keras_layers__global_max_pool1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pool1_d",
        "_map_keras_layers__global_max_pool1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pool1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pool1_d",
        "_map_keras_layers__global_max_pool1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pool1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pool2_d() -> None:
    """Verify _map_keras_layers__global_max_pool2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pool2_d",
        "_map_keras_layers__global_max_pool2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pool2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pool2_d",
        "_map_keras_layers__global_max_pool2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pool2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pool3_d() -> None:
    """Verify _map_keras_layers__global_max_pool3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pool3_d",
        "_map_keras_layers__global_max_pool3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pool3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pool3_d",
        "_map_keras_layers__global_max_pool3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pool3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pooling1_d() -> None:
    """Verify _map_keras_layers__global_max_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pooling1_d",
        "_map_keras_layers__global_max_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pooling1_d",
        "_map_keras_layers__global_max_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pooling2_d() -> None:
    """Verify _map_keras_layers__global_max_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pooling2_d",
        "_map_keras_layers__global_max_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pooling2_d",
        "_map_keras_layers__global_max_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__global_max_pooling3_d() -> None:
    """Verify _map_keras_layers__global_max_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__global_max_pooling3_d",
        "_map_keras_layers__global_max_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__global_max_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__global_max_pooling3_d",
        "_map_keras_layers__global_max_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__global_max_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pool1_d() -> None:
    """Verify _map_keras_layers__max_pool1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pool1_d",
        "_map_keras_layers__max_pool1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pool1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pool1_d", "_map_keras_layers__max_pool1_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__max_pool1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pool2_d() -> None:
    """Verify _map_keras_layers__max_pool2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pool2_d",
        "_map_keras_layers__max_pool2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pool2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pool2_d", "_map_keras_layers__max_pool2_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__max_pool2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pool3_d() -> None:
    """Verify _map_keras_layers__max_pool3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pool3_d",
        "_map_keras_layers__max_pool3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pool3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pool3_d", "_map_keras_layers__max_pool3_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__max_pool3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pooling1_d() -> None:
    """Verify _map_keras_layers__max_pooling1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pooling1_d",
        "_map_keras_layers__max_pooling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pooling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pooling1_d",
        "_map_keras_layers__max_pooling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__max_pooling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pooling2_d() -> None:
    """Verify _map_keras_layers__max_pooling2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pooling2_d",
        "_map_keras_layers__max_pooling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pooling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pooling2_d",
        "_map_keras_layers__max_pooling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__max_pooling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_pooling3_d() -> None:
    """Verify _map_keras_layers__max_pooling3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_pooling3_d",
        "_map_keras_layers__max_pooling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_pooling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_pooling3_d",
        "_map_keras_layers__max_pooling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__max_pooling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_max_pooling_1d() -> None:
    """Verify _map_keras_max_pooling_1d emits MaxPool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_max_pooling_1d",
        "_map_keras_max_pooling_1d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_max_pooling_1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MaxPool"

    node_empty = TFNode(
        "empty__map_keras_max_pooling_1d", "_map_keras_max_pooling_1d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_max_pooling_1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_max_pooling_2d() -> None:
    """Verify _map_keras_max_pooling_2d emits MaxPool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_max_pooling_2d",
        "_map_keras_max_pooling_2d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_max_pooling_2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MaxPool"

    node_empty = TFNode(
        "empty__map_keras_max_pooling_2d", "_map_keras_max_pooling_2d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_max_pooling_2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_max_pooling_3d() -> None:
    """Verify _map_keras_max_pooling_3d emits MaxPool with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_max_pooling_3d",
        "_map_keras_max_pooling_3d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_max_pooling_3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MaxPool"

    node_empty = TFNode(
        "empty__map_keras_max_pooling_3d", "_map_keras_max_pooling_3d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_max_pooling_3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_batch_normalization() -> None:
    """Verify _map_keras_batch_normalization emits BatchNormalization with valid inputs and BatchNormalization when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_batch_normalization",
        "_map_keras_batch_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_batch_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "BatchNormalization"

    node_empty = TFNode(
        "empty__map_keras_batch_normalization", "_map_keras_batch_normalization", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_batch_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "BatchNormalization"


def test__map_keras_constraints() -> None:
    """Verify _map_keras_constraints emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints", "_map_keras_constraints", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_constraints(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints", "_map_keras_constraints", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_constraints(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints__constraint() -> None:
    """Verify _map_keras_constraints__constraint emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints__constraint",
        "_map_keras_constraints__constraint",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints__constraint(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints__constraint",
        "_map_keras_constraints__constraint",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints__constraint(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints__max_norm() -> None:
    """Verify _map_keras_constraints__max_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints__max_norm",
        "_map_keras_constraints__max_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints__max_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints__max_norm",
        "_map_keras_constraints__max_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints__max_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints__min_max_norm() -> None:
    """Verify _map_keras_constraints__min_max_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints__min_max_norm",
        "_map_keras_constraints__min_max_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints__min_max_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints__min_max_norm",
        "_map_keras_constraints__min_max_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints__min_max_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints__non_neg() -> None:
    """Verify _map_keras_constraints__non_neg emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints__non_neg",
        "_map_keras_constraints__non_neg",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints__non_neg(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints__non_neg",
        "_map_keras_constraints__non_neg",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints__non_neg(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints__unit_norm() -> None:
    """Verify _map_keras_constraints__unit_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints__unit_norm",
        "_map_keras_constraints__unit_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints__unit_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints__unit_norm",
        "_map_keras_constraints__unit_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints__unit_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_deserialize() -> None:
    """Verify _map_keras_constraints_deserialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_deserialize",
        "_map_keras_constraints_deserialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_deserialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_deserialize",
        "_map_keras_constraints_deserialize",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints_deserialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_get() -> None:
    """Verify _map_keras_constraints_get emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_get",
        "_map_keras_constraints_get",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_get(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_get", "_map_keras_constraints_get", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_constraints_get(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_max_norm() -> None:
    """Verify _map_keras_constraints_max_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_max_norm",
        "_map_keras_constraints_max_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_max_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_max_norm",
        "_map_keras_constraints_max_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints_max_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_min_max_norm() -> None:
    """Verify _map_keras_constraints_min_max_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_min_max_norm",
        "_map_keras_constraints_min_max_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_min_max_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_min_max_norm",
        "_map_keras_constraints_min_max_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints_min_max_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_non_neg() -> None:
    """Verify _map_keras_constraints_non_neg emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_non_neg",
        "_map_keras_constraints_non_neg",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_non_neg(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_non_neg", "_map_keras_constraints_non_neg", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_constraints_non_neg(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_serialize() -> None:
    """Verify _map_keras_constraints_serialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_serialize",
        "_map_keras_constraints_serialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_serialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_serialize",
        "_map_keras_constraints_serialize",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints_serialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_constraints_unit_norm() -> None:
    """Verify _map_keras_constraints_unit_norm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_constraints_unit_norm",
        "_map_keras_constraints_unit_norm",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_constraints_unit_norm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_constraints_unit_norm",
        "_map_keras_constraints_unit_norm",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_constraints_unit_norm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_dropout() -> None:
    """Verify _map_keras_dropout emits Dropout with valid inputs and Dropout when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_dropout", "_map_keras_dropout", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_dropout(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Dropout"

    node_empty = TFNode("empty__map_keras_dropout", "_map_keras_dropout", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_dropout(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Dropout"


def test__map_keras_gaussian_dropout() -> None:
    """Verify _map_keras_gaussian_dropout emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_gaussian_dropout",
        "_map_keras_gaussian_dropout",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_gaussian_dropout(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_gaussian_dropout", "_map_keras_gaussian_dropout", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_gaussian_dropout(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_gaussian_noise() -> None:
    """Verify _map_keras_gaussian_noise emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_gaussian_noise",
        "_map_keras_gaussian_noise",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_gaussian_noise(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_gaussian_noise", "_map_keras_gaussian_noise", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_gaussian_noise(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layer_normalization() -> None:
    """Verify _map_keras_layer_normalization emits LayerNormalization with valid inputs and LayerNormalization when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layer_normalization",
        "_map_keras_layer_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layer_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "LayerNormalization"

    node_empty = TFNode(
        "empty__map_keras_layer_normalization", "_map_keras_layer_normalization", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layer_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "LayerNormalization"


def test__map_keras_layers__activity_regularization() -> None:
    """Verify _map_keras_layers__activity_regularization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__activity_regularization",
        "_map_keras_layers__activity_regularization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__activity_regularization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__activity_regularization",
        "_map_keras_layers__activity_regularization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__activity_regularization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__alpha_dropout() -> None:
    """Verify _map_keras_layers__alpha_dropout emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__alpha_dropout",
        "_map_keras_layers__alpha_dropout",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__alpha_dropout(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__alpha_dropout",
        "_map_keras_layers__alpha_dropout",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__alpha_dropout(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__batch_normalization() -> None:
    """Verify _map_keras_layers__batch_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__batch_normalization",
        "_map_keras_layers__batch_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__batch_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__batch_normalization",
        "_map_keras_layers__batch_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__batch_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__dropout() -> None:
    """Verify _map_keras_layers__dropout emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__dropout",
        "_map_keras_layers__dropout",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__dropout(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__dropout", "_map_keras_layers__dropout", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__dropout(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__gaussian_dropout() -> None:
    """Verify _map_keras_layers__gaussian_dropout emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__gaussian_dropout",
        "_map_keras_layers__gaussian_dropout",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__gaussian_dropout(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__gaussian_dropout",
        "_map_keras_layers__gaussian_dropout",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__gaussian_dropout(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__gaussian_noise() -> None:
    """Verify _map_keras_layers__gaussian_noise emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__gaussian_noise",
        "_map_keras_layers__gaussian_noise",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__gaussian_noise(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__gaussian_noise",
        "_map_keras_layers__gaussian_noise",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__gaussian_noise(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__group_normalization() -> None:
    """Verify _map_keras_layers__group_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__group_normalization",
        "_map_keras_layers__group_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__group_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__group_normalization",
        "_map_keras_layers__group_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__group_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__layer_normalization() -> None:
    """Verify _map_keras_layers__layer_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__layer_normalization",
        "_map_keras_layers__layer_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__layer_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__layer_normalization",
        "_map_keras_layers__layer_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__layer_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__spatial_dropout1_d() -> None:
    """Verify _map_keras_layers__spatial_dropout1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__spatial_dropout1_d",
        "_map_keras_layers__spatial_dropout1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__spatial_dropout1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__spatial_dropout1_d",
        "_map_keras_layers__spatial_dropout1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__spatial_dropout1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__spatial_dropout2_d() -> None:
    """Verify _map_keras_layers__spatial_dropout2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__spatial_dropout2_d",
        "_map_keras_layers__spatial_dropout2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__spatial_dropout2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__spatial_dropout2_d",
        "_map_keras_layers__spatial_dropout2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__spatial_dropout2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__spatial_dropout3_d() -> None:
    """Verify _map_keras_layers__spatial_dropout3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__spatial_dropout3_d",
        "_map_keras_layers__spatial_dropout3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__spatial_dropout3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__spatial_dropout3_d",
        "_map_keras_layers__spatial_dropout3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__spatial_dropout3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__spectral_normalization() -> None:
    """Verify _map_keras_layers__spectral_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__spectral_normalization",
        "_map_keras_layers__spectral_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__spectral_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__spectral_normalization",
        "_map_keras_layers__spectral_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__spectral_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__unit_normalization() -> None:
    """Verify _map_keras_layers__unit_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__unit_normalization",
        "_map_keras_layers__unit_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__unit_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__unit_normalization",
        "_map_keras_layers__unit_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__unit_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_rms_normalization() -> None:
    """Verify _map_keras_layers_rms_normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_rms_normalization",
        "_map_keras_layers_rms_normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_rms_normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_rms_normalization",
        "_map_keras_layers_rms_normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers_rms_normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_bidirectional() -> None:
    """Verify _map_keras_bidirectional emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_bidirectional", "_map_keras_bidirectional", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_bidirectional(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode(
        "empty__map_keras_bidirectional", "_map_keras_bidirectional", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_bidirectional(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv_lstm1d() -> None:
    """Verify _map_keras_conv_lstm1d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv_lstm1d", "_map_keras_conv_lstm1d", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_conv_lstm1d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_conv_lstm1d", "_map_keras_conv_lstm1d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv_lstm1d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv_lstm2d() -> None:
    """Verify _map_keras_conv_lstm2d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv_lstm2d", "_map_keras_conv_lstm2d", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_conv_lstm2d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_conv_lstm2d", "_map_keras_conv_lstm2d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv_lstm2d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_conv_lstm3d() -> None:
    """Verify _map_keras_conv_lstm3d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_conv_lstm3d", "_map_keras_conv_lstm3d", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_conv_lstm3d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_conv_lstm3d", "_map_keras_conv_lstm3d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_conv_lstm3d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_gru() -> None:
    """Verify _map_keras_gru emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_gru", "_map_keras_gru", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_gru(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode("empty__map_keras_gru", "_map_keras_gru", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_gru(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__bidirectional() -> None:
    """Verify _map_keras_layers__bidirectional emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__bidirectional",
        "_map_keras_layers__bidirectional",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__bidirectional(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode(
        "empty__map_keras_layers__bidirectional",
        "_map_keras_layers__bidirectional",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__bidirectional(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv_lstm1_d() -> None:
    """Verify _map_keras_layers__conv_lstm1_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv_lstm1_d",
        "_map_keras_layers__conv_lstm1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv_lstm1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv_lstm1_d",
        "_map_keras_layers__conv_lstm1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv_lstm1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv_lstm2_d() -> None:
    """Verify _map_keras_layers__conv_lstm2_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv_lstm2_d",
        "_map_keras_layers__conv_lstm2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv_lstm2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv_lstm2_d",
        "_map_keras_layers__conv_lstm2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv_lstm2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__conv_lstm3_d() -> None:
    """Verify _map_keras_layers__conv_lstm3_d emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__conv_lstm3_d",
        "_map_keras_layers__conv_lstm3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__conv_lstm3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__conv_lstm3_d",
        "_map_keras_layers__conv_lstm3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__conv_lstm3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__simple_rnn() -> None:
    """Verify _map_keras_layers__simple_rnn emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__simple_rnn",
        "_map_keras_layers__simple_rnn",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__simple_rnn(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode(
        "empty__map_keras_layers__simple_rnn", "_map_keras_layers__simple_rnn", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__simple_rnn(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__simple_rnn_cell() -> None:
    """Verify _map_keras_layers__simple_rnn_cell emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__simple_rnn_cell",
        "_map_keras_layers__simple_rnn_cell",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__simple_rnn_cell(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode(
        "empty__map_keras_layers__simple_rnn_cell",
        "_map_keras_layers__simple_rnn_cell",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__simple_rnn_cell(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__stacked_rnn_cells() -> None:
    """Verify _map_keras_layers__stacked_rnn_cells emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__stacked_rnn_cells",
        "_map_keras_layers__stacked_rnn_cells",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__stacked_rnn_cells(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__stacked_rnn_cells",
        "_map_keras_layers__stacked_rnn_cells",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__stacked_rnn_cells(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__time_distributed() -> None:
    """Verify _map_keras_layers__time_distributed emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__time_distributed",
        "_map_keras_layers__time_distributed",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__time_distributed(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__time_distributed",
        "_map_keras_layers__time_distributed",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__time_distributed(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_gru() -> None:
    """Verify _map_keras_layers_gru emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_gru", "_map_keras_layers_gru", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_gru(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_layers_gru", "_map_keras_layers_gru", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_layers_gru(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_gru_cell() -> None:
    """Verify _map_keras_layers_gru_cell emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_gru_cell",
        "_map_keras_layers_gru_cell",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_gru_cell(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_gru_cell", "_map_keras_layers_gru_cell", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_gru_cell(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_lstm() -> None:
    """Verify _map_keras_layers_lstm emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_lstm", "_map_keras_layers_lstm", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_lstm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_lstm", "_map_keras_layers_lstm", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_lstm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_lstm_cell() -> None:
    """Verify _map_keras_layers_lstm_cell emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_lstm_cell",
        "_map_keras_layers_lstm_cell",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_lstm_cell(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_lstm_cell", "_map_keras_layers_lstm_cell", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_lstm_cell(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_rnn() -> None:
    """Verify _map_keras_layers_rnn emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_rnn", "_map_keras_layers_rnn", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_rnn(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_layers_rnn", "_map_keras_layers_rnn", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_layers_rnn(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_lstm() -> None:
    """Verify _map_keras_lstm emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_lstm", "_map_keras_lstm", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_lstm(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode("empty__map_keras_lstm", "_map_keras_lstm", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_lstm(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_rnn() -> None:
    """Verify _map_keras_rnn emits RNN with valid inputs and RNN when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_rnn", "_map_keras_rnn", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_rnn(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "RNN"

    node_empty = TFNode("empty__map_keras_rnn", "_map_keras_rnn", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_rnn(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "RNN"


def test__map_keras_simple_rnn() -> None:
    """Verify _map_keras_simple_rnn emits Squeeze with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_simple_rnn", "_map_keras_simple_rnn", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_simple_rnn(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Squeeze"

    node_empty = TFNode("empty__map_keras_simple_rnn", "_map_keras_simple_rnn", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_simple_rnn(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_time_distributed() -> None:
    """Verify _map_keras_time_distributed emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_time_distributed",
        "_map_keras_time_distributed",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_time_distributed(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_time_distributed", "_map_keras_time_distributed", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_time_distributed(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_attention_base() -> None:
    """Verify _map_keras_attention_base emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_attention_base",
        "_map_keras_attention_base",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_attention_base(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_attention_base", "_map_keras_attention_base", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_attention_base(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__additive_attention() -> None:
    """Verify _map_keras_layers__additive_attention emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__additive_attention",
        "_map_keras_layers__additive_attention",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__additive_attention(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__additive_attention",
        "_map_keras_layers__additive_attention",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__additive_attention(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__attention() -> None:
    """Verify _map_keras_layers__attention emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__attention",
        "_map_keras_layers__attention",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__attention(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__attention", "_map_keras_layers__attention", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__attention(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__group_query_attention() -> None:
    """Verify _map_keras_layers__group_query_attention emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__group_query_attention",
        "_map_keras_layers__group_query_attention",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__group_query_attention(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__group_query_attention",
        "_map_keras_layers__group_query_attention",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__group_query_attention(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__multi_head_attention() -> None:
    """Verify _map_keras_layers__multi_head_attention emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__multi_head_attention",
        "_map_keras_layers__multi_head_attention",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__multi_head_attention(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__multi_head_attention",
        "_map_keras_layers__multi_head_attention",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__multi_head_attention(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_multi_head_attention() -> None:
    """Verify _map_keras_multi_head_attention emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_multi_head_attention",
        "_map_keras_multi_head_attention",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_multi_head_attention(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_multi_head_attention",
        "_map_keras_multi_head_attention",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_multi_head_attention(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activation() -> None:
    """Verify _map_keras_activation emits Identity with valid inputs and Identity when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activation", "_map_keras_activation", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_activation(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_activation", "_map_keras_activation", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_activation(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"


def test__map_keras_activations_celu() -> None:
    """Verify _map_keras_activations_celu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_celu",
        "_map_keras_activations_celu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_celu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_celu", "_map_keras_activations_celu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_celu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_deserialize() -> None:
    """Verify _map_keras_activations_deserialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_deserialize",
        "_map_keras_activations_deserialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_deserialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_deserialize",
        "_map_keras_activations_deserialize",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_deserialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_elu() -> None:
    """Verify _map_keras_activations_elu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_elu",
        "_map_keras_activations_elu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_elu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_elu", "_map_keras_activations_elu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_elu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_exponential() -> None:
    """Verify _map_keras_activations_exponential emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_exponential",
        "_map_keras_activations_exponential",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_exponential(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_exponential",
        "_map_keras_activations_exponential",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_exponential(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_gelu() -> None:
    """Verify _map_keras_activations_gelu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_gelu",
        "_map_keras_activations_gelu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_gelu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_gelu", "_map_keras_activations_gelu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_gelu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_get() -> None:
    """Verify _map_keras_activations_get emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_get",
        "_map_keras_activations_get",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_get(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_get", "_map_keras_activations_get", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_get(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_glu() -> None:
    """Verify _map_keras_activations_glu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_glu",
        "_map_keras_activations_glu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_glu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_glu", "_map_keras_activations_glu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_glu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_hard_shrink() -> None:
    """Verify _map_keras_activations_hard_shrink emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_hard_shrink",
        "_map_keras_activations_hard_shrink",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_hard_shrink(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_hard_shrink",
        "_map_keras_activations_hard_shrink",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_hard_shrink(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_hard_sigmoid() -> None:
    """Verify _map_keras_activations_hard_sigmoid emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_hard_sigmoid",
        "_map_keras_activations_hard_sigmoid",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_hard_sigmoid(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_hard_sigmoid",
        "_map_keras_activations_hard_sigmoid",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_hard_sigmoid(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_hard_silu() -> None:
    """Verify _map_keras_activations_hard_silu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_hard_silu",
        "_map_keras_activations_hard_silu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_hard_silu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_hard_silu",
        "_map_keras_activations_hard_silu",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_hard_silu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_hard_swish() -> None:
    """Verify _map_keras_activations_hard_swish emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_hard_swish",
        "_map_keras_activations_hard_swish",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_hard_swish(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_hard_swish",
        "_map_keras_activations_hard_swish",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_hard_swish(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_hard_tanh() -> None:
    """Verify _map_keras_activations_hard_tanh emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_hard_tanh",
        "_map_keras_activations_hard_tanh",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_hard_tanh(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_hard_tanh",
        "_map_keras_activations_hard_tanh",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_hard_tanh(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_leaky_relu() -> None:
    """Verify _map_keras_activations_leaky_relu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_leaky_relu",
        "_map_keras_activations_leaky_relu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_leaky_relu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_leaky_relu",
        "_map_keras_activations_leaky_relu",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_leaky_relu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_linear() -> None:
    """Verify _map_keras_activations_linear emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_linear",
        "_map_keras_activations_linear",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_linear(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_linear", "_map_keras_activations_linear", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_linear(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_log_sigmoid() -> None:
    """Verify _map_keras_activations_log_sigmoid emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_log_sigmoid",
        "_map_keras_activations_log_sigmoid",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_log_sigmoid(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_log_sigmoid",
        "_map_keras_activations_log_sigmoid",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_log_sigmoid(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_log_softmax() -> None:
    """Verify _map_keras_activations_log_softmax emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_log_softmax",
        "_map_keras_activations_log_softmax",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_log_softmax(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_log_softmax",
        "_map_keras_activations_log_softmax",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_log_softmax(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_mish() -> None:
    """Verify _map_keras_activations_mish emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_mish",
        "_map_keras_activations_mish",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_mish(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_mish", "_map_keras_activations_mish", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_mish(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_relu() -> None:
    """Verify _map_keras_activations_relu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_relu",
        "_map_keras_activations_relu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_relu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_relu", "_map_keras_activations_relu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_relu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_relu6() -> None:
    """Verify _map_keras_activations_relu6 emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_relu6",
        "_map_keras_activations_relu6",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_relu6(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_relu6", "_map_keras_activations_relu6", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_relu6(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_selu() -> None:
    """Verify _map_keras_activations_selu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_selu",
        "_map_keras_activations_selu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_selu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_selu", "_map_keras_activations_selu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_selu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_serialize() -> None:
    """Verify _map_keras_activations_serialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_serialize",
        "_map_keras_activations_serialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_serialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_serialize",
        "_map_keras_activations_serialize",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_serialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_sigmoid() -> None:
    """Verify _map_keras_activations_sigmoid emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_sigmoid",
        "_map_keras_activations_sigmoid",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_sigmoid(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_sigmoid", "_map_keras_activations_sigmoid", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_sigmoid(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_silu() -> None:
    """Verify _map_keras_activations_silu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_silu",
        "_map_keras_activations_silu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_silu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_silu", "_map_keras_activations_silu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_silu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_soft_shrink() -> None:
    """Verify _map_keras_activations_soft_shrink emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_soft_shrink",
        "_map_keras_activations_soft_shrink",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_soft_shrink(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_soft_shrink",
        "_map_keras_activations_soft_shrink",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_soft_shrink(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_softmax() -> None:
    """Verify _map_keras_activations_softmax emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_softmax",
        "_map_keras_activations_softmax",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_softmax(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_softmax", "_map_keras_activations_softmax", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_softmax(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_softplus() -> None:
    """Verify _map_keras_activations_softplus emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_softplus",
        "_map_keras_activations_softplus",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_softplus(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_softplus",
        "_map_keras_activations_softplus",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_softplus(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_softsign() -> None:
    """Verify _map_keras_activations_softsign emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_softsign",
        "_map_keras_activations_softsign",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_softsign(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_softsign",
        "_map_keras_activations_softsign",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_softsign(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_sparse_plus() -> None:
    """Verify _map_keras_activations_sparse_plus emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_sparse_plus",
        "_map_keras_activations_sparse_plus",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_sparse_plus(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_sparse_plus",
        "_map_keras_activations_sparse_plus",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_sparse_plus(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_sparse_sigmoid() -> None:
    """Verify _map_keras_activations_sparse_sigmoid emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_sparse_sigmoid",
        "_map_keras_activations_sparse_sigmoid",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_sparse_sigmoid(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_sparse_sigmoid",
        "_map_keras_activations_sparse_sigmoid",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_sparse_sigmoid(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_sparsemax() -> None:
    """Verify _map_keras_activations_sparsemax emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_sparsemax",
        "_map_keras_activations_sparsemax",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_sparsemax(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_sparsemax",
        "_map_keras_activations_sparsemax",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_sparsemax(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_squareplus() -> None:
    """Verify _map_keras_activations_squareplus emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_squareplus",
        "_map_keras_activations_squareplus",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_squareplus(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_squareplus",
        "_map_keras_activations_squareplus",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_squareplus(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_swish() -> None:
    """Verify _map_keras_activations_swish emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_swish",
        "_map_keras_activations_swish",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_swish(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_swish", "_map_keras_activations_swish", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_swish(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_tanh() -> None:
    """Verify _map_keras_activations_tanh emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_tanh",
        "_map_keras_activations_tanh",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_tanh(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_tanh", "_map_keras_activations_tanh", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_activations_tanh(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_tanh_shrink() -> None:
    """Verify _map_keras_activations_tanh_shrink emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_tanh_shrink",
        "_map_keras_activations_tanh_shrink",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_tanh_shrink(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_tanh_shrink",
        "_map_keras_activations_tanh_shrink",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_tanh_shrink(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_activations_threshold() -> None:
    """Verify _map_keras_activations_threshold emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_activations_threshold",
        "_map_keras_activations_threshold",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_activations_threshold(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_activations_threshold",
        "_map_keras_activations_threshold",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_activations_threshold(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__activation() -> None:
    """Verify _map_keras_layers__activation emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__activation",
        "_map_keras_layers__activation",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__activation(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__activation", "_map_keras_layers__activation", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__activation(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__leaky_re_lu() -> None:
    """Verify _map_keras_layers__leaky_re_lu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__leaky_re_lu",
        "_map_keras_layers__leaky_re_lu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__leaky_re_lu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__leaky_re_lu", "_map_keras_layers__leaky_re_lu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__leaky_re_lu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__re_lu() -> None:
    """Verify _map_keras_layers__re_lu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__re_lu", "_map_keras_layers__re_lu", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__re_lu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__re_lu", "_map_keras_layers__re_lu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__re_lu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__softmax() -> None:
    """Verify _map_keras_layers__softmax emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__softmax",
        "_map_keras_layers__softmax",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__softmax(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__softmax", "_map_keras_layers__softmax", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__softmax(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_elu() -> None:
    """Verify _map_keras_layers_elu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_elu", "_map_keras_layers_elu", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_elu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_layers_elu", "_map_keras_layers_elu", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_layers_elu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_p_re_lu() -> None:
    """Verify _map_keras_layers_p_re_lu emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_p_re_lu",
        "_map_keras_layers_p_re_lu",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_p_re_lu(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_p_re_lu", "_map_keras_layers_p_re_lu", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_p_re_lu(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_add() -> None:
    """Verify _map_keras_add emits Sum with valid inputs and Sum when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_add", "_map_keras_add", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_add(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Sum"

    node_empty = TFNode("empty__map_keras_add", "_map_keras_add", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_add(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Sum"


def test__map_keras_average() -> None:
    """Verify _map_keras_average emits Mean with valid inputs and Mean when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_average", "_map_keras_average", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_average(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Mean"

    node_empty = TFNode("empty__map_keras_average", "_map_keras_average", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_average(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Mean"


def test__map_keras_concatenate() -> None:
    """Verify _map_keras_concatenate emits Concat with valid inputs and Concat when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_concatenate", "_map_keras_concatenate", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_concatenate(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Concat"

    node_empty = TFNode(
        "empty__map_keras_concatenate", "_map_keras_concatenate", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_concatenate(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Concat"


def test__map_keras_dot() -> None:
    """Verify _map_keras_dot emits MatMul with valid inputs and MatMul when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_dot", "_map_keras_dot", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_dot(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MatMul"

    node_empty = TFNode("empty__map_keras_dot", "_map_keras_dot", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_dot(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "MatMul"


def test__map_keras_layers__add() -> None:
    """Verify _map_keras_layers__add emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__add", "_map_keras_layers__add", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__add(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__add", "_map_keras_layers__add", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__add(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__average() -> None:
    """Verify _map_keras_layers__average emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__average",
        "_map_keras_layers__average",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__average(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__average", "_map_keras_layers__average", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__average(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__concatenate() -> None:
    """Verify _map_keras_layers__concatenate emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__concatenate",
        "_map_keras_layers__concatenate",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__concatenate(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__concatenate", "_map_keras_layers__concatenate", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__concatenate(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__dot() -> None:
    """Verify _map_keras_layers__dot emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__dot", "_map_keras_layers__dot", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__dot(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__dot", "_map_keras_layers__dot", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__dot(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__maximum() -> None:
    """Verify _map_keras_layers__maximum emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__maximum",
        "_map_keras_layers__maximum",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__maximum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__maximum", "_map_keras_layers__maximum", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__maximum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__minimum() -> None:
    """Verify _map_keras_layers__minimum emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__minimum",
        "_map_keras_layers__minimum",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__minimum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__minimum", "_map_keras_layers__minimum", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__minimum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__multiply() -> None:
    """Verify _map_keras_layers__multiply emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__multiply",
        "_map_keras_layers__multiply",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__multiply(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__multiply", "_map_keras_layers__multiply", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__multiply(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__subtract() -> None:
    """Verify _map_keras_layers__subtract emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__subtract",
        "_map_keras_layers__subtract",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__subtract(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__subtract", "_map_keras_layers__subtract", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__subtract(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__zero_padding1_d() -> None:
    """Verify _map_keras_layers__zero_padding1_d emits Pad with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__zero_padding1_d",
        "_map_keras_layers__zero_padding1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__zero_padding1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode(
        "empty__map_keras_layers__zero_padding1_d",
        "_map_keras_layers__zero_padding1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__zero_padding1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__zero_padding2_d() -> None:
    """Verify _map_keras_layers__zero_padding2_d emits Pad with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__zero_padding2_d",
        "_map_keras_layers__zero_padding2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__zero_padding2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode(
        "empty__map_keras_layers__zero_padding2_d",
        "_map_keras_layers__zero_padding2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__zero_padding2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__zero_padding3_d() -> None:
    """Verify _map_keras_layers__zero_padding3_d emits Pad with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__zero_padding3_d",
        "_map_keras_layers__zero_padding3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__zero_padding3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode(
        "empty__map_keras_layers__zero_padding3_d",
        "_map_keras_layers__zero_padding3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__zero_padding3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_add() -> None:
    """Verify _map_keras_layers_add emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_add", "_map_keras_layers_add", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_add(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_layers_add", "_map_keras_layers_add", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_layers_add(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_average() -> None:
    """Verify _map_keras_layers_average emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_average",
        "_map_keras_layers_average",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_average(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_average", "_map_keras_layers_average", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_average(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_concatenate() -> None:
    """Verify _map_keras_layers_concatenate emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_concatenate",
        "_map_keras_layers_concatenate",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_concatenate(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_concatenate", "_map_keras_layers_concatenate", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_concatenate(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_dot() -> None:
    """Verify _map_keras_layers_dot emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_dot", "_map_keras_layers_dot", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers_dot(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_layers_dot", "_map_keras_layers_dot", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_layers_dot(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_maximum() -> None:
    """Verify _map_keras_layers_maximum emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_maximum",
        "_map_keras_layers_maximum",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_maximum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_maximum", "_map_keras_layers_maximum", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_maximum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_minimum() -> None:
    """Verify _map_keras_layers_minimum emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_minimum",
        "_map_keras_layers_minimum",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_minimum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_minimum", "_map_keras_layers_minimum", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_minimum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_multiply() -> None:
    """Verify _map_keras_layers_multiply emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_multiply",
        "_map_keras_layers_multiply",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_multiply(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_multiply", "_map_keras_layers_multiply", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_multiply(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_subtract() -> None:
    """Verify _map_keras_layers_subtract emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_subtract",
        "_map_keras_layers_subtract",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_subtract(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_subtract", "_map_keras_layers_subtract", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_subtract(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_maximum() -> None:
    """Verify _map_keras_maximum emits Max with valid inputs and Max when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_maximum", "_map_keras_maximum", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_maximum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Max"

    node_empty = TFNode("empty__map_keras_maximum", "_map_keras_maximum", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_maximum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Max"


def test__map_keras_minimum() -> None:
    """Verify _map_keras_minimum emits Min with valid inputs and Min when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_minimum", "_map_keras_minimum", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_minimum(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Min"

    node_empty = TFNode("empty__map_keras_minimum", "_map_keras_minimum", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_minimum(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Min"


def test__map_keras_multiply() -> None:
    """Verify _map_keras_multiply emits Mul with valid inputs and Mul when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_multiply", "_map_keras_multiply", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_multiply(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Mul"

    node_empty = TFNode("empty__map_keras_multiply", "_map_keras_multiply", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_multiply(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Mul"


def test__map_keras_subtract() -> None:
    """Verify _map_keras_subtract emits Sub with valid inputs and Sub when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_subtract", "_map_keras_subtract", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_subtract(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Sub"

    node_empty = TFNode("empty__map_keras_subtract", "_map_keras_subtract", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_subtract(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Sub"


def test__map_keras_dense() -> None:
    """Verify _map_keras_dense emits Add with valid inputs and MatMul when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_dense", "_map_keras_dense", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_dense(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Add"

    node_empty = TFNode("empty__map_keras_dense", "_map_keras_dense", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_dense(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "MatMul"


def test__map_keras_einsum_dense() -> None:
    """Verify _map_keras_einsum_dense emits Add with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_einsum_dense", "_map_keras_einsum_dense", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_einsum_dense(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Add"

    node_empty = TFNode(
        "empty__map_keras_einsum_dense", "_map_keras_einsum_dense", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_einsum_dense(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_embedding() -> None:
    """Verify _map_keras_embedding emits Gather with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_embedding", "_map_keras_embedding", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_embedding(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Gather"

    node_empty = TFNode("empty__map_keras_embedding", "_map_keras_embedding", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_embedding(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_flatten() -> None:
    """Verify _map_keras_flatten emits Flatten with valid inputs and Flatten when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_flatten", "_map_keras_flatten", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_flatten(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Flatten"

    node_empty = TFNode("empty__map_keras_flatten", "_map_keras_flatten", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_flatten(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Flatten"


def test__map_keras_layers__cropping1_d() -> None:
    """Verify _map_keras_layers__cropping1_d emits Slice with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__cropping1_d",
        "_map_keras_layers__cropping1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__cropping1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode(
        "empty__map_keras_layers__cropping1_d", "_map_keras_layers__cropping1_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__cropping1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__cropping2_d() -> None:
    """Verify _map_keras_layers__cropping2_d emits Slice with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__cropping2_d",
        "_map_keras_layers__cropping2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__cropping2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode(
        "empty__map_keras_layers__cropping2_d", "_map_keras_layers__cropping2_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__cropping2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__cropping3_d() -> None:
    """Verify _map_keras_layers__cropping3_d emits Slice with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__cropping3_d",
        "_map_keras_layers__cropping3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__cropping3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode(
        "empty__map_keras_layers__cropping3_d", "_map_keras_layers__cropping3_d", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__cropping3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__dense() -> None:
    """Verify _map_keras_layers__dense emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__dense", "_map_keras_layers__dense", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__dense(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__dense", "_map_keras_layers__dense", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__dense(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__einsum_dense() -> None:
    """Verify _map_keras_layers__einsum_dense emits Add with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__einsum_dense",
        "_map_keras_layers__einsum_dense",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__einsum_dense(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Add"

    node_empty = TFNode(
        "empty__map_keras_layers__einsum_dense",
        "_map_keras_layers__einsum_dense",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__einsum_dense(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__embedding() -> None:
    """Verify _map_keras_layers__embedding emits Gather with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__embedding",
        "_map_keras_layers__embedding",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__embedding(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Gather"

    node_empty = TFNode(
        "empty__map_keras_layers__embedding", "_map_keras_layers__embedding", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__embedding(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__flatten() -> None:
    """Verify _map_keras_layers__flatten emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__flatten",
        "_map_keras_layers__flatten",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__flatten(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__flatten", "_map_keras_layers__flatten", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__flatten(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__flax_layer() -> None:
    """Verify _map_keras_layers__flax_layer emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__flax_layer",
        "_map_keras_layers__flax_layer",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__flax_layer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__flax_layer", "_map_keras_layers__flax_layer", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__flax_layer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__identity() -> None:
    """Verify _map_keras_layers__identity emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__identity",
        "_map_keras_layers__identity",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__identity(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__identity", "_map_keras_layers__identity", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__identity(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__input() -> None:
    """Verify _map_keras_layers__input emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__input", "_map_keras_layers__input", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__input(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__input", "_map_keras_layers__input", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__input(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__input_layer() -> None:
    """Verify _map_keras_layers__input_layer emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__input_layer",
        "_map_keras_layers__input_layer",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__input_layer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__input_layer", "_map_keras_layers__input_layer", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__input_layer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__input_spec() -> None:
    """Verify _map_keras_layers__input_spec emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__input_spec",
        "_map_keras_layers__input_spec",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__input_spec(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__input_spec", "_map_keras_layers__input_spec", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__input_spec(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__jax_layer() -> None:
    """Verify _map_keras_layers__jax_layer emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__jax_layer",
        "_map_keras_layers__jax_layer",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__jax_layer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__jax_layer", "_map_keras_layers__jax_layer", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__jax_layer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__lambda() -> None:
    """Verify _map_keras_layers__lambda emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__lambda",
        "_map_keras_layers__lambda",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__lambda(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__lambda", "_map_keras_layers__lambda", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__lambda(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__layer() -> None:
    """Verify _map_keras_layers__layer emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__layer", "_map_keras_layers__layer", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_layers__layer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__layer", "_map_keras_layers__layer", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__layer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__masking() -> None:
    """Verify _map_keras_layers__masking emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__masking",
        "_map_keras_layers__masking",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__masking(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__masking", "_map_keras_layers__masking", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__masking(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__permute() -> None:
    """Verify _map_keras_layers__permute emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__permute",
        "_map_keras_layers__permute",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__permute(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__permute", "_map_keras_layers__permute", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__permute(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__repeat_vector() -> None:
    """Verify _map_keras_layers__repeat_vector emits Tile with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__repeat_vector",
        "_map_keras_layers__repeat_vector",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__repeat_vector(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Tile"

    node_empty = TFNode(
        "empty__map_keras_layers__repeat_vector",
        "_map_keras_layers__repeat_vector",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__repeat_vector(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__reshape() -> None:
    """Verify _map_keras_layers__reshape emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__reshape",
        "_map_keras_layers__reshape",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__reshape(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__reshape", "_map_keras_layers__reshape", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__reshape(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__reversible_embedding() -> None:
    """Verify _map_keras_layers__reversible_embedding emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__reversible_embedding",
        "_map_keras_layers__reversible_embedding",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__reversible_embedding(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__reversible_embedding",
        "_map_keras_layers__reversible_embedding",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__reversible_embedding(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__torch_module_wrapper() -> None:
    """Verify _map_keras_layers__torch_module_wrapper emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__torch_module_wrapper",
        "_map_keras_layers__torch_module_wrapper",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__torch_module_wrapper(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__torch_module_wrapper",
        "_map_keras_layers__torch_module_wrapper",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__torch_module_wrapper(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__up_sampling1_d() -> None:
    """Verify _map_keras_layers__up_sampling1_d emits Resize with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__up_sampling1_d",
        "_map_keras_layers__up_sampling1_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__up_sampling1_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode(
        "empty__map_keras_layers__up_sampling1_d",
        "_map_keras_layers__up_sampling1_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__up_sampling1_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__up_sampling2_d() -> None:
    """Verify _map_keras_layers__up_sampling2_d emits Resize with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__up_sampling2_d",
        "_map_keras_layers__up_sampling2_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__up_sampling2_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode(
        "empty__map_keras_layers__up_sampling2_d",
        "_map_keras_layers__up_sampling2_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__up_sampling2_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__up_sampling3_d() -> None:
    """Verify _map_keras_layers__up_sampling3_d emits Resize with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__up_sampling3_d",
        "_map_keras_layers__up_sampling3_d",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__up_sampling3_d(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode(
        "empty__map_keras_layers__up_sampling3_d",
        "_map_keras_layers__up_sampling3_d",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__up_sampling3_d(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__wrapper() -> None:
    """Verify _map_keras_layers__wrapper emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__wrapper",
        "_map_keras_layers__wrapper",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__wrapper(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__wrapper", "_map_keras_layers__wrapper", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__wrapper(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_deserialize() -> None:
    """Verify _map_keras_layers_deserialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_deserialize",
        "_map_keras_layers_deserialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_deserialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_deserialize", "_map_keras_layers_deserialize", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_deserialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_serialize() -> None:
    """Verify _map_keras_layers_serialize emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_serialize",
        "_map_keras_layers_serialize",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_serialize(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_serialize", "_map_keras_layers_serialize", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_serialize(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_tfsm_layer() -> None:
    """Verify _map_keras_layers_tfsm_layer emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_tfsm_layer",
        "_map_keras_layers_tfsm_layer",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_tfsm_layer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers_tfsm_layer", "_map_keras_layers_tfsm_layer", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers_tfsm_layer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_masking() -> None:
    """Verify _map_keras_masking emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_masking", "_map_keras_masking", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_masking(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_masking", "_map_keras_masking", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_masking(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_permute() -> None:
    """Verify _map_keras_permute emits Transpose with valid inputs and Transpose when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_permute", "_map_keras_permute", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_permute(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Transpose"

    node_empty = TFNode("empty__map_keras_permute", "_map_keras_permute", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_permute(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Transpose"


def test__map_keras_repeat_vector() -> None:
    """Verify _map_keras_repeat_vector emits Tile with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_repeat_vector", "_map_keras_repeat_vector", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_repeat_vector(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Tile"

    node_empty = TFNode(
        "empty__map_keras_repeat_vector", "_map_keras_repeat_vector", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_repeat_vector(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_reshape() -> None:
    """Verify _map_keras_reshape emits Reshape with valid inputs and Reshape when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_reshape", "_map_keras_reshape", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_reshape(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Reshape"

    node_empty = TFNode("empty__map_keras_reshape", "_map_keras_reshape", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_reshape(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Reshape"


def test__map_keras_hashing() -> None:
    """Verify _map_keras_hashing emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_hashing", "_map_keras_hashing", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_hashing(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode("empty__map_keras_hashing", "_map_keras_hashing", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_hashing(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_initializers_stft() -> None:
    """Verify _map_keras_initializers_stft emits STFT with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_initializers_stft",
        "_map_keras_initializers_stft",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_initializers_stft(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode(
        "empty__map_keras_initializers_stft", "_map_keras_initializers_stft", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_initializers_stft(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_initializers_stft_ext() -> None:
    """Verify _map_keras_initializers_stft_ext emits STFT with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_initializers_stft_ext",
        "_map_keras_initializers_stft_ext",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_initializers_stft_ext(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode(
        "empty__map_keras_initializers_stft_ext",
        "_map_keras_initializers_stft_ext",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_initializers_stft_ext(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_initializers_stft_initializer() -> None:
    """Verify _map_keras_initializers_stft_initializer emits STFT with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_initializers_stft_initializer",
        "_map_keras_initializers_stft_initializer",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_initializers_stft_initializer(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode(
        "empty__map_keras_initializers_stft_initializer",
        "_map_keras_initializers_stft_initializer",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_initializers_stft_initializer(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_integer_lookup() -> None:
    """Verify _map_keras_integer_lookup emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_integer_lookup",
        "_map_keras_integer_lookup",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_integer_lookup(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode(
        "empty__map_keras_integer_lookup", "_map_keras_integer_lookup", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_integer_lookup(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__aug_mix() -> None:
    """Verify _map_keras_layers__aug_mix emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__aug_mix",
        "_map_keras_layers__aug_mix",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__aug_mix(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__aug_mix", "_map_keras_layers__aug_mix", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__aug_mix(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__auto_contrast() -> None:
    """Verify _map_keras_layers__auto_contrast emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__auto_contrast",
        "_map_keras_layers__auto_contrast",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__auto_contrast(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__auto_contrast",
        "_map_keras_layers__auto_contrast",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__auto_contrast(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__category_encoding() -> None:
    """Verify _map_keras_layers__category_encoding emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__category_encoding",
        "_map_keras_layers__category_encoding",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__category_encoding(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__category_encoding",
        "_map_keras_layers__category_encoding",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__category_encoding(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__center_crop() -> None:
    """Verify _map_keras_layers__center_crop emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__center_crop",
        "_map_keras_layers__center_crop",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__center_crop(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__center_crop", "_map_keras_layers__center_crop", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__center_crop(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__cut_mix() -> None:
    """Verify _map_keras_layers__cut_mix emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__cut_mix",
        "_map_keras_layers__cut_mix",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__cut_mix(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__cut_mix", "_map_keras_layers__cut_mix", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__cut_mix(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__discretization() -> None:
    """Verify _map_keras_layers__discretization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__discretization",
        "_map_keras_layers__discretization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__discretization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__discretization",
        "_map_keras_layers__discretization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__discretization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__equalization() -> None:
    """Verify _map_keras_layers__equalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__equalization",
        "_map_keras_layers__equalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__equalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__equalization",
        "_map_keras_layers__equalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__equalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__hashed_crossing() -> None:
    """Verify _map_keras_layers__hashed_crossing emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__hashed_crossing",
        "_map_keras_layers__hashed_crossing",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__hashed_crossing(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__hashed_crossing",
        "_map_keras_layers__hashed_crossing",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__hashed_crossing(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__hashing() -> None:
    """Verify _map_keras_layers__hashing emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__hashing",
        "_map_keras_layers__hashing",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__hashing(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode(
        "empty__map_keras_layers__hashing", "_map_keras_layers__hashing", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__hashing(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__integer_lookup() -> None:
    """Verify _map_keras_layers__integer_lookup emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__integer_lookup",
        "_map_keras_layers__integer_lookup",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__integer_lookup(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode(
        "empty__map_keras_layers__integer_lookup",
        "_map_keras_layers__integer_lookup",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__integer_lookup(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__max_num_bounding_boxes() -> None:
    """Verify _map_keras_layers__max_num_bounding_boxes emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__max_num_bounding_boxes",
        "_map_keras_layers__max_num_bounding_boxes",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__max_num_bounding_boxes(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__max_num_bounding_boxes",
        "_map_keras_layers__max_num_bounding_boxes",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__max_num_bounding_boxes(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__mel_spectrogram() -> None:
    """Verify _map_keras_layers__mel_spectrogram emits MatMul with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__mel_spectrogram",
        "_map_keras_layers__mel_spectrogram",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__mel_spectrogram(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MatMul"

    node_empty = TFNode(
        "empty__map_keras_layers__mel_spectrogram",
        "_map_keras_layers__mel_spectrogram",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__mel_spectrogram(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__mix_up() -> None:
    """Verify _map_keras_layers__mix_up emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__mix_up",
        "_map_keras_layers__mix_up",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__mix_up(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__mix_up", "_map_keras_layers__mix_up", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__mix_up(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__normalization() -> None:
    """Verify _map_keras_layers__normalization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__normalization",
        "_map_keras_layers__normalization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__normalization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__normalization",
        "_map_keras_layers__normalization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__normalization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__pipeline() -> None:
    """Verify _map_keras_layers__pipeline emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__pipeline",
        "_map_keras_layers__pipeline",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__pipeline(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__pipeline", "_map_keras_layers__pipeline", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__pipeline(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__rand_augment() -> None:
    """Verify _map_keras_layers__rand_augment emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__rand_augment",
        "_map_keras_layers__rand_augment",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__rand_augment(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__rand_augment",
        "_map_keras_layers__rand_augment",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__rand_augment(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_brightness() -> None:
    """Verify _map_keras_layers__random_brightness emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_brightness",
        "_map_keras_layers__random_brightness",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_brightness(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_brightness",
        "_map_keras_layers__random_brightness",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_brightness(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_color_degeneration() -> None:
    """Verify _map_keras_layers__random_color_degeneration emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_color_degeneration",
        "_map_keras_layers__random_color_degeneration",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_color_degeneration(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_color_degeneration",
        "_map_keras_layers__random_color_degeneration",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_color_degeneration(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_color_jitter() -> None:
    """Verify _map_keras_layers__random_color_jitter emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_color_jitter",
        "_map_keras_layers__random_color_jitter",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_color_jitter(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_color_jitter",
        "_map_keras_layers__random_color_jitter",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_color_jitter(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_contrast() -> None:
    """Verify _map_keras_layers__random_contrast emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_contrast",
        "_map_keras_layers__random_contrast",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_contrast(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_contrast",
        "_map_keras_layers__random_contrast",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_contrast(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_crop() -> None:
    """Verify _map_keras_layers__random_crop emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_crop",
        "_map_keras_layers__random_crop",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_crop(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_crop", "_map_keras_layers__random_crop", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__random_crop(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_elastic_transform() -> None:
    """Verify _map_keras_layers__random_elastic_transform emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_elastic_transform",
        "_map_keras_layers__random_elastic_transform",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_elastic_transform(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_elastic_transform",
        "_map_keras_layers__random_elastic_transform",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_elastic_transform(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_erasing() -> None:
    """Verify _map_keras_layers__random_erasing emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_erasing",
        "_map_keras_layers__random_erasing",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_erasing(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_erasing",
        "_map_keras_layers__random_erasing",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_erasing(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_flip() -> None:
    """Verify _map_keras_layers__random_flip emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_flip",
        "_map_keras_layers__random_flip",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_flip(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_flip", "_map_keras_layers__random_flip", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__random_flip(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_gaussian_blur() -> None:
    """Verify _map_keras_layers__random_gaussian_blur emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_gaussian_blur",
        "_map_keras_layers__random_gaussian_blur",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_gaussian_blur(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_gaussian_blur",
        "_map_keras_layers__random_gaussian_blur",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_gaussian_blur(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_grayscale() -> None:
    """Verify _map_keras_layers__random_grayscale emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_grayscale",
        "_map_keras_layers__random_grayscale",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_grayscale(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_grayscale",
        "_map_keras_layers__random_grayscale",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_grayscale(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_hue() -> None:
    """Verify _map_keras_layers__random_hue emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_hue",
        "_map_keras_layers__random_hue",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_hue(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_hue", "_map_keras_layers__random_hue", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__random_hue(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_invert() -> None:
    """Verify _map_keras_layers__random_invert emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_invert",
        "_map_keras_layers__random_invert",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_invert(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_invert",
        "_map_keras_layers__random_invert",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_invert(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_perspective() -> None:
    """Verify _map_keras_layers__random_perspective emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_perspective",
        "_map_keras_layers__random_perspective",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_perspective(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_perspective",
        "_map_keras_layers__random_perspective",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_perspective(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_posterization() -> None:
    """Verify _map_keras_layers__random_posterization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_posterization",
        "_map_keras_layers__random_posterization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_posterization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_posterization",
        "_map_keras_layers__random_posterization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_posterization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_rotation() -> None:
    """Verify _map_keras_layers__random_rotation emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_rotation",
        "_map_keras_layers__random_rotation",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_rotation(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_rotation",
        "_map_keras_layers__random_rotation",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_rotation(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_saturation() -> None:
    """Verify _map_keras_layers__random_saturation emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_saturation",
        "_map_keras_layers__random_saturation",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_saturation(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_saturation",
        "_map_keras_layers__random_saturation",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_saturation(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_sharpness() -> None:
    """Verify _map_keras_layers__random_sharpness emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_sharpness",
        "_map_keras_layers__random_sharpness",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_sharpness(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_sharpness",
        "_map_keras_layers__random_sharpness",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_sharpness(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_shear() -> None:
    """Verify _map_keras_layers__random_shear emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_shear",
        "_map_keras_layers__random_shear",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_shear(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_shear",
        "_map_keras_layers__random_shear",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_shear(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_translation() -> None:
    """Verify _map_keras_layers__random_translation emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_translation",
        "_map_keras_layers__random_translation",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_translation(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_translation",
        "_map_keras_layers__random_translation",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__random_translation(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__random_zoom() -> None:
    """Verify _map_keras_layers__random_zoom emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__random_zoom",
        "_map_keras_layers__random_zoom",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__random_zoom(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__random_zoom", "_map_keras_layers__random_zoom", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__random_zoom(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__rescaling() -> None:
    """Verify _map_keras_layers__rescaling emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__rescaling",
        "_map_keras_layers__rescaling",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__rescaling(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__rescaling", "_map_keras_layers__rescaling", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__rescaling(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__resizing() -> None:
    """Verify _map_keras_layers__resizing emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__resizing",
        "_map_keras_layers__resizing",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__resizing(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__resizing", "_map_keras_layers__resizing", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_layers__resizing(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__solarization() -> None:
    """Verify _map_keras_layers__solarization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__solarization",
        "_map_keras_layers__solarization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__solarization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__solarization",
        "_map_keras_layers__solarization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__solarization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__string_lookup() -> None:
    """Verify _map_keras_layers__string_lookup emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__string_lookup",
        "_map_keras_layers__string_lookup",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__string_lookup(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode(
        "empty__map_keras_layers__string_lookup",
        "_map_keras_layers__string_lookup",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__string_lookup(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers__text_vectorization() -> None:
    """Verify _map_keras_layers__text_vectorization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers__text_vectorization",
        "_map_keras_layers__text_vectorization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers__text_vectorization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_layers__text_vectorization",
        "_map_keras_layers__text_vectorization",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers__text_vectorization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_layers_stft_spectrogram() -> None:
    """Verify _map_keras_layers_stft_spectrogram emits STFT with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_layers_stft_spectrogram",
        "_map_keras_layers_stft_spectrogram",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_layers_stft_spectrogram(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode(
        "empty__map_keras_layers_stft_spectrogram",
        "_map_keras_layers_stft_spectrogram",
        inputs=[],
        attr={},
    )
    outputs_empty = keras_layers._map_keras_layers_stft_spectrogram(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_mel_spectrogram() -> None:
    """Verify _map_keras_mel_spectrogram emits MatMul with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_mel_spectrogram",
        "_map_keras_mel_spectrogram",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_mel_spectrogram(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "MatMul"

    node_empty = TFNode(
        "empty__map_keras_mel_spectrogram", "_map_keras_mel_spectrogram", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_mel_spectrogram(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_mfcc() -> None:
    """Verify _map_keras_mfcc emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_mfcc", "_map_keras_mfcc", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_mfcc(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("empty__map_keras_mfcc", "_map_keras_mfcc", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_mfcc(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_stft() -> None:
    """Verify _map_keras_stft emits STFT with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode("node__map_keras_stft", "_map_keras_stft", inputs=["x", "w", "b"], attr={})
    outputs = keras_layers._map_keras_stft(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode("empty__map_keras_stft", "_map_keras_stft", inputs=[], attr={})
    outputs_empty = keras_layers._map_keras_stft(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_string_lookup() -> None:
    """Verify _map_keras_string_lookup emits CategoryMapper with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_string_lookup", "_map_keras_string_lookup", inputs=["x", "w", "b"], attr={}
    )
    outputs = keras_layers._map_keras_string_lookup(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode(
        "empty__map_keras_string_lookup", "_map_keras_string_lookup", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_string_lookup(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"


def test__map_keras_text_vectorization() -> None:
    """Verify _map_keras_text_vectorization emits Identity with valid inputs and Constant when inputs are empty."""
    builder = TFToONNXGraphBuilder("test")
    node = TFNode(
        "node__map_keras_text_vectorization",
        "_map_keras_text_vectorization",
        inputs=["x", "w", "b"],
        attr={},
    )
    outputs = keras_layers._map_keras_text_vectorization(builder, node)
    assert len(outputs) >= 1
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode(
        "empty__map_keras_text_vectorization", "_map_keras_text_vectorization", inputs=[], attr={}
    )
    outputs_empty = keras_layers._map_keras_text_vectorization(builder, node_empty)
    assert len(outputs_empty) >= 1
    assert builder.graph.nodes[-1].op_type == "Constant"
