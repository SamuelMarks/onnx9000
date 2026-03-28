"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn.containers import (
    ModuleDict,
    ModuleList,
    ParameterList,
    Sequential,
)
from onnx9000.converters.frontend.nn.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
)
from onnx9000.converters.frontend.nn.dropout import Dropout, Dropout2d, Dropout3d
from onnx9000.converters.frontend.nn.embedding import Embedding
from onnx9000.converters.frontend.nn.flatten import Flatten, Unflatten
from onnx9000.converters.frontend.nn.functional import (
    conv2d,
    gelu,
    interpolate,
    linear,
    one_hot,
    pad,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from onnx9000.converters.frontend.nn.identity import Identity
from onnx9000.converters.frontend.nn.linear import Linear
from onnx9000.converters.frontend.nn.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from onnx9000.converters.frontend.nn.pool import (
    AdaptiveAvgPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)
from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


def test_containers() -> None:
    """Tests the test_containers functionality."""
    s = Sequential(Identity(), Identity())
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1,), DType.FLOAT32)
        s(t)
    ml = ModuleList([Identity()])
    ml.append(Identity())
    ml.extend([Identity()])
    for _ in ml:
        pass
    md = ModuleDict({"a": Identity()})
    md["b"] = Identity()
    md.update({"c": Identity()})


def test_conv() -> None:
    """Tests the test_conv functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        c1 = Conv1d(3, 6, 3, bias=False)
        c1(Tensor((1, 3, 10), DType.FLOAT32))
        c2 = Conv2d(3, 6, 3, padding=1, dilation=2)
        c2(Tensor((1, 3, 10, 10), DType.FLOAT32))
        c3 = Conv3d(3, 6, 3)
        c3(Tensor((1, 3, 10, 10, 10), DType.FLOAT32))
        ct1 = ConvTranspose1d(3, 6, 3, bias=False)
        ct1(Tensor((1, 3, 10), DType.FLOAT32))
        ct2 = ConvTranspose2d(3, 6, 3, output_padding=1)
        ct2(Tensor((1, 3, 10, 10), DType.FLOAT32))


def test_dropout() -> None:
    """Tests the test_dropout functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        d1 = Dropout(0.5)
        d2 = Dropout2d(0.5)
        d3 = Dropout3d(0.5)
        d1(t)
        d2(t)
        d3(t)


def test_embedding() -> None:
    """Tests the test_embedding functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        e = Embedding(10, 5)
        e(Tensor((1, 10), DType.INT64))


def test_flatten() -> None:
    """Tests the test_flatten functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        f = Flatten(1)
        f(t)
        u = Unflatten(1, (3, 10, 10))
        u(Tensor((1, 300), DType.FLOAT32))


def test_functional() -> None:
    """Tests the test_functional functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        relu(t)
        sigmoid(t)
        tanh(t)
        gelu(t)
        softmax(t, dim=1)
        w = Tensor((5, 10), DType.FLOAT32)
        t_l = Tensor((1, 10), DType.FLOAT32)
        linear(t_l, w)
        w_c = Tensor((6, 3, 3, 3), DType.FLOAT32)
        conv2d(t, w_c)
        pad(t, (1, 1, 1, 1))
        interpolate(t, size=(20, 20))
        t_i = Tensor((1, 10), DType.INT64)
        one_hot(t_i, 5)


def test_identity() -> None:
    """Tests the test_identity functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        i = Identity()
        i(Tensor((1,), DType.FLOAT32))


def test_linear() -> None:
    """Tests the test_linear functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        l1 = Linear(10, 5)
        l1(Tensor((1, 10), DType.FLOAT32))
        l2 = Linear(10, 5, bias=False)
        l2(Tensor((1, 10), DType.FLOAT32))


def test_normalization() -> None:
    """Tests the test_normalization functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        b1 = BatchNorm1d(3, affine=False)
        b1(Tensor((1, 3, 10), DType.FLOAT32))
        b2 = BatchNorm2d(3)
        b2(Tensor((1, 3, 10, 10), DType.FLOAT32))
        b3 = BatchNorm3d(3)
        b3(Tensor((1, 3, 10, 10, 10), DType.FLOAT32))
        ln = LayerNorm(10)
        ln(Tensor((1, 10), DType.FLOAT32))
        ln2 = LayerNorm(10, elementwise_affine=False)
        ln2(Tensor((1, 10), DType.FLOAT32))
        gn = GroupNorm(2, 6)
        gn(Tensor((1, 6, 10, 10), DType.FLOAT32))
        gn2 = GroupNorm(2, 6, affine=False)
        gn2(Tensor((1, 6, 10, 10), DType.FLOAT32))
        in1 = InstanceNorm1d(3)
        in1(Tensor((1, 3, 10), DType.FLOAT32))
        in2 = InstanceNorm2d(3, affine=True)
        in2(Tensor((1, 3, 10, 10), DType.FLOAT32))
        in3 = InstanceNorm3d(3)
        in3(Tensor((1, 3, 10, 10, 10), DType.FLOAT32))


def test_pool() -> None:
    """Tests the test_pool functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        m1 = MaxPool1d(2)
        m1(Tensor((1, 3, 10), DType.FLOAT32))
        m2 = MaxPool2d(2)
        m2(Tensor((1, 3, 10, 10), DType.FLOAT32))
        a1 = AvgPool1d(2)
        a1(Tensor((1, 3, 10), DType.FLOAT32))
        a2 = AvgPool2d(2)
        a2(Tensor((1, 3, 10, 10), DType.FLOAT32))
        aa2 = AdaptiveAvgPool2d((1, 1))
        aa2(Tensor((1, 3, 10, 10), DType.FLOAT32))


def test_rnn() -> None:
    """Tests the test_rnn functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        r = RNN(10, 5, 2, bias=False, bidirectional=True)
        r(Tensor((1, 10, 10), DType.FLOAT32))
        l = LSTM(10, 5)
        l(Tensor((1, 10, 10), DType.FLOAT32))
        g = GRU(10, 5)
        g(Tensor((1, 10, 10), DType.FLOAT32))


def test_parameter_list() -> None:
    """Tests the test_parameter_list functionality."""
    pl = ParameterList([Parameter((1,), DType.FLOAT32, "p1")])
    pl.append(Parameter((1,), DType.FLOAT32, "p2"))
    pl.extend([Parameter((1,), DType.FLOAT32, "p3")])
    assert len(pl) == 3
    _ = pl[0]
    _ = pl[-1]
    for _p in pl:
        pass


def test_moduledict_errors() -> None:
    """Tests the test_moduledict_errors functionality."""
    with pytest.raises(TypeError):
        ModuleDict(123)


def test_sequential_dict() -> None:
    """Tests the test_sequential_dict functionality."""
    from collections import OrderedDict

    d = OrderedDict()
    d["i"] = Identity()
    s = Sequential(d)
    assert len(list(s.children())) == 1


def test_modulelist_slice() -> None:
    """Tests the test_modulelist_slice functionality."""
    ml = ModuleList([Identity(), Identity()])
    _ = ml[0:1]
    _ = ml[-1]


def test_functional_kwargs() -> None:
    """Tests the test_functional_kwargs functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        from onnx9000.converters.frontend.nn.functional import interpolate, pad

        interpolate(t, scale_factor=2.0)
        pad(t, (1, 1), value=1.0)


def test_init_functions() -> None:
    """Tests the init functions functionality."""
    from onnx9000.converters.frontend.nn import init

    t = Tensor(data=np.ones((10, 10), dtype=np.float32))
    init.constant_(t, 5.0)
    init.ones_(t)
    init.zeros_(t)
    init.xavier_uniform_(t)
    init.xavier_normal_(t)
    init.kaiming_uniform_(t)
    init.kaiming_normal_(t)
    init.kaiming_uniform_(t, mode="fan_out")
    init.kaiming_normal_(t, mode="fan_out")
    t3 = Tensor(data=np.ones((10, 10, 3, 3), dtype=np.float32))
    init.calculate_fan_in_and_fan_out(t3)
    with pytest.raises(ValueError):
        init.calculate_fan_in_and_fan_out(Tensor((10,), DType.FLOAT32))


def test_conv_repr() -> None:
    """Tests the test_conv_repr functionality."""
    c = Conv1d(3, 6, 3)
    _ = repr(c)
    c2 = Conv2d(3, 6, 3)
    _ = repr(c2)
    c3 = Conv3d(3, 6, 3)
    _ = repr(c3)
    ct1 = ConvTranspose1d(3, 6, 3)
    _ = repr(ct1)
    ct2 = ConvTranspose2d(3, 6, 3)
    _ = repr(ct2)


def test_pool_repr() -> None:
    """Tests the test_pool_repr functionality."""
    p1 = MaxPool1d(2)
    _ = repr(p1)
    p2 = MaxPool2d(2)
    _ = repr(p2)


def test_norm_repr() -> None:
    """Tests the test_norm_repr functionality."""
    b = BatchNorm1d(3)
    _ = repr(b)
    l = LayerNorm(10)
    _ = repr(l)


def test_dropout_repr() -> None:
    """Tests the test_dropout_repr functionality."""
    d = Dropout()
    _ = repr(d)
    d2 = Dropout2d()
    _ = repr(d2)


def test_rnn_repr() -> None:
    """Tests the test_rnn_repr functionality."""
    r = RNN(10, 5)
    _ = repr(r)


def test_moduledict_methods() -> None:
    """Tests the test_moduledict_methods functionality."""
    md = ModuleDict({"a": Identity()})
    md["b"] = Identity()
    md.update({"c": Identity()})
    assert len(list(md.keys())) == 3
    assert len(list(md.values())) == 3
    assert len(list(md.items())) == 3
    assert isinstance(md["a"], Identity)
    md["d"] = Identity()


def test_functional_missing_coverage() -> None:
    """Tests the test_functional_missing_coverage functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        w = Tensor((6, 3, 3, 3), DType.FLOAT32)
        b = Tensor((6,), DType.FLOAT32)
        from onnx9000.converters.frontend.nn.functional import conv2d, linear

        t_l = Tensor((1, 10), DType.FLOAT32)
        w_l = Tensor((5, 10), DType.FLOAT32)
        b_l = Tensor((5,), DType.FLOAT32)
        linear(t_l, w_l, bias=b_l)
        conv2d(t, w, bias=b, stride=(2, 2), padding=(1, 1), dilation=(2, 2))
        from onnx9000.converters.frontend.nn.functional import interpolate, pad

        interpolate(t, scale_factor=2.0, mode="linear")
        pad(t, (1, 1))


def test_normalization_missing() -> None:
    """Tests the test_normalization_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        BatchNorm1d(3, track_running_stats=False)
        BatchNorm2d(3, track_running_stats=False)
        BatchNorm3d(3, track_running_stats=False)
        InstanceNorm2d(3, track_running_stats=True)
        InstanceNorm3d(3, affine=True)


def test_rnn_missing() -> None:
    """Tests the test_rnn_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        LSTM(10, 5, bidirectional=True)
        GRU(10, 5, bidirectional=True)


def test_dropout_missing() -> None:
    """Tests the test_dropout_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        d1 = Dropout(p=0)
        d1.eval()
        d1(Tensor((10,), DType.FLOAT32))
        d2 = Dropout2d(p=0)
        d2.eval()
        d2(Tensor((10,), DType.FLOAT32))
        d3 = Dropout3d(p=0)
        d3.eval()
        d3(Tensor((10,), DType.FLOAT32))


def test_conv_missing() -> None:
    """Tests the test_conv_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        Conv1d(3, 6, 3, padding="same")
        Conv2d(3, 6, 3, padding="same")
        Conv3d(3, 6, 3, padding="same")


def test_pool_missing() -> None:
    """Tests the test_pool_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        MaxPool1d(2)
        MaxPool2d(2)


def test_pool_tuple() -> None:
    """Tests the test_pool_tuple functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        m1 = MaxPool1d((2,))
        m1(Tensor((1, 3, 10), DType.FLOAT32))
        m2 = MaxPool2d((2, 2))
        m2(Tensor((1, 3, 10, 10), DType.FLOAT32))


def test_conv_bias_false() -> None:
    """Tests the test_conv_bias_false functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        Conv2d(3, 6, 3, bias=False)
        Conv3d(3, 6, 3, bias=False)
        ConvTranspose2d(3, 6, 3, bias=False)


def test_functional_interpolate_tuple() -> None:
    """Tests the test_functional_interpolate_tuple functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 10, 10), DType.FLOAT32)
        from onnx9000.converters.frontend.nn.functional import interpolate

        interpolate(t, scale_factor=(2.0, 2.0))


def test_rnn_list_return() -> None:
    """Tests the test_rnn_list_return functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        from unittest.mock import patch

        def fake_rec(op, inp, attrs=None):
            """Test the fake_rec functionality."""
            return [Tensor((1,), DType.FLOAT32)]

        with patch("onnx9000.converters.frontend.utils.record_op", fake_rec):
            d = Dropout(0.5)
            d(Tensor((1,), DType.FLOAT32))
        with patch("onnx9000.converters.frontend.utils.record_op", fake_rec):
            n = BatchNorm1d(3)
            n(Tensor((1, 3, 10), DType.FLOAT32))
        with patch("onnx9000.converters.frontend.utils.record_op", fake_rec):
            r = RNN(10, 5)
            r(Tensor((1, 10, 10), DType.FLOAT32))
            l = LSTM(10, 5)
            l(Tensor((1, 10, 10), DType.FLOAT32))
            g = GRU(10, 5)
            g(Tensor((1, 10, 10), DType.FLOAT32))


def test_rnn_proj_size() -> None:
    """Tests the test_rnn_proj_size functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        l = LSTM(10, 5, batch_first=True)
        l(Tensor((1, 10, 10), DType.FLOAT32))


def test_rnn_hx_and_bias() -> None:
    """Tests the test_rnn_hx_and_bias functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        hx = Tensor((1, 10, 5), DType.FLOAT32)
        hc = Tensor((1, 10, 5), DType.FLOAT32)
        r = RNN(10, 5, bias=True)
        r(Tensor((1, 10, 10), DType.FLOAT32), hx)
        l = LSTM(10, 5, bias=False)
        l(Tensor((1, 10, 10), DType.FLOAT32), (hx, hc))
        g = GRU(10, 5, bias=True)
        g(Tensor((1, 10, 10), DType.FLOAT32), hx)


def test_dropout_coverage_23() -> None:
    """Tests the test_dropout_coverage_23 functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        d = Dropout(0.5)
        d(Tensor((1,), DType.FLOAT32))


def test_rnn_list_missing() -> None:
    """Tests the test_rnn_list_missing functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        from unittest.mock import patch

        def fake_rec(op, inp, attrs=None):
            """Test the fake_rec functionality."""
            return Tensor((1,), DType.FLOAT32)

        with patch("onnx9000.converters.frontend.utils.record_op", fake_rec):
            r = RNN(10, 5)
            r(Tensor((1, 10, 10), DType.FLOAT32))
            l = LSTM(10, 5)
            l(Tensor((1, 10, 10), DType.FLOAT32))
            g = GRU(10, 5)
            g(Tensor((1, 10, 10), DType.FLOAT32))
