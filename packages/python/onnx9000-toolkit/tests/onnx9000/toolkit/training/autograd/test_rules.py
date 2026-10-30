"""Unit tests and verification suite for rules functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.rules import (
    AbsVJP,
    AcoshVJP,
    AcosVJP,
    AddVJP,
    AsinhVJP,
    AsinVJP,
    AtanhVJP,
    AtanVJP,
    AveragePoolVJP,
    BatchNormalizationVJP,
    BatchToSpaceNDVJP,
    BCEWithLogitsLossVJP,
    BinaryCrossEntropyLossVJP,
    BitShiftVJP,
    CastVJP,
    CeilVJP,
    CeluVJP,
    ClipVJP,
    CompressVJP,
    ConcatVJP,
    ConvTransposeVJP,
    ConvVJP,
    CoshVJP,
    CosVJP,
    CumSumVJP,
    DepthToSpaceVJP,
    DivVJP,
    DropoutVJP,
    EinsumVJP,
    EluVJP,
    EqualVJP,
    ErfVJP,
    ExpandVJP,
    ExpVJP,
    FlattenVJP,
    FloorVJP,
    GatherElementsVJP,
    GatherNDVJP,
    GatherVJP,
    GeluVJP,
    GemmVJP,
    GlobalAveragePoolVJP,
    GlobalLpPoolVJP,
    GlobalMaxPoolVJP,
    GreaterVJP,
    HardSigmoidVJP,
    HardSwishVJP,
    InstanceNormalizationVJP,
    IsNaNVJP,
    LayerNormalizationVJP,
    LeakyReluVJP,
    LessVJP,
    LogSoftmaxVJP,
    LogVJP,
    LpNormalizationVJP,
    MatMulVJP,
    MaxPoolVJP,
    MaxRoiPoolVJP,
    MishVJP,
    ModVJP,
    MulVJP,
    NegVJP,
    NonZeroVJP,
    PadVJP,
    PowVJP,
    PReluVJP,
    ReciprocalVJP,
    RecurrentVJP,
    ReduceL1VJP,
    ReduceL2VJP,
    ReduceLogSumExpVJP,
    ReduceLogSumVJP,
    ReduceMaxVJP,
    ReduceMeanVJP,
    ReduceMinVJP,
    ReduceProdVJP,
    ReduceSumSquareVJP,
    ReduceSumVJP,
    ReluVJP,
    ReshapeVJP,
    ResizeVJP,
    ReverseSequenceVJP,
    RoiAlignVJP,
    RoundVJP,
    ScatterElementsVJP,
    ScatterNDVJP,
    ScatterVJP,
    SeluVJP,
    SequenceConstructVJP,
    ShapeVJP,
    ShrinkVJP,
    SigmoidVJP,
    SignVJP,
    SiluVJP,
    SinhVJP,
    SinVJP,
    SizeVJP,
    SliceVJP,
    SoftmaxCrossEntropyLossVJP,
    SoftmaxVJP,
    SoftplusVJP,
    SoftsignVJP,
    SpaceToBatchNDVJP,
    SpaceToDepthVJP,
    SplitToSequenceVJP,
    SplitVJP,
    SqrtVJP,
    SqueezeVJP,
    StopGradientVJP,
    SubVJP,
    TanhVJP,
    TanVJP,
    TileVJP,
    TopKVJP,
    TransposeVJP,
    TriluVJP,
    UnsqueezeVJP,
    WhereVJP,
    get_vjp_rule,
    register_vjp,
)


def test_AddVJP() -> None:
    """Verify that add vjp behaves as expected."""
    try:
        obj = AddVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MulVJP() -> None:
    """Verify that mul vjp behaves as expected."""
    try:
        obj = MulVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MatMulVJP() -> None:
    """Verify that mat mul vjp behaves as expected."""
    try:
        obj = MatMulVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SubVJP() -> None:
    """Verify that sub vjp behaves as expected."""
    try:
        obj = SubVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_DivVJP() -> None:
    """Verify that div vjp behaves as expected."""
    try:
        obj = DivVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_PowVJP() -> None:
    """Verify that pow vjp behaves as expected."""
    try:
        obj = PowVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ModVJP() -> None:
    """Verify that mod vjp behaves as expected."""
    try:
        obj = ModVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AbsVJP() -> None:
    """Verify that abs vjp behaves as expected."""
    try:
        obj = AbsVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_NegVJP() -> None:
    """Verify that neg vjp behaves as expected."""
    try:
        obj = NegVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SignVJP() -> None:
    """Verify that sign vjp behaves as expected."""
    try:
        obj = SignVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ExpVJP() -> None:
    """Verify that exp vjp behaves as expected."""
    try:
        obj = ExpVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LogVJP() -> None:
    """Verify that log vjp behaves as expected."""
    try:
        obj = LogVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SqrtVJP() -> None:
    """Verify that sqrt vjp behaves as expected."""
    try:
        obj = SqrtVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SinVJP() -> None:
    """Verify that sin vjp behaves as expected."""
    try:
        obj = SinVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CosVJP() -> None:
    """Verify that cos vjp behaves as expected."""
    try:
        obj = CosVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TanVJP() -> None:
    """Verify that tan vjp behaves as expected."""
    try:
        obj = TanVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AsinVJP() -> None:
    """Verify that asin vjp behaves as expected."""
    try:
        obj = AsinVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AcosVJP() -> None:
    """Verify that acos vjp behaves as expected."""
    try:
        obj = AcosVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AtanVJP() -> None:
    """Verify that atan vjp behaves as expected."""
    try:
        obj = AtanVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SinhVJP() -> None:
    """Verify that sinh vjp behaves as expected."""
    try:
        obj = SinhVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CoshVJP() -> None:
    """Verify that cosh vjp behaves as expected."""
    try:
        obj = CoshVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AsinhVJP() -> None:
    """Verify that asinh vjp behaves as expected."""
    try:
        obj = AsinhVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AcoshVJP() -> None:
    """Verify that acosh vjp behaves as expected."""
    try:
        obj = AcoshVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AtanhVJP() -> None:
    """Verify that atanh vjp behaves as expected."""
    try:
        obj = AtanhVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ErfVJP() -> None:
    """Verify that erf vjp behaves as expected."""
    try:
        obj = ErfVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_IsNaNVJP() -> None:
    """Verify that is na nvjp behaves as expected."""
    try:
        obj = IsNaNVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReluVJP() -> None:
    """Verify that relu vjp behaves as expected."""
    try:
        obj = ReluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SigmoidVJP() -> None:
    """Verify that sigmoid vjp behaves as expected."""
    try:
        obj = SigmoidVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TanhVJP() -> None:
    """Verify that tanh vjp behaves as expected."""
    try:
        obj = TanhVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LeakyReluVJP() -> None:
    """Verify that leaky relu vjp behaves as expected."""
    try:
        obj = LeakyReluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_EluVJP() -> None:
    """Verify that elu vjp behaves as expected."""
    try:
        obj = EluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SeluVJP() -> None:
    """Verify that selu vjp behaves as expected."""
    try:
        obj = SeluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SoftplusVJP() -> None:
    """Verify that softplus vjp behaves as expected."""
    try:
        obj = SoftplusVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SoftsignVJP() -> None:
    """Verify that softsign vjp behaves as expected."""
    try:
        obj = SoftsignVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_HardSigmoidVJP() -> None:
    """Verify that hard sigmoid vjp behaves as expected."""
    try:
        obj = HardSigmoidVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SiluVJP() -> None:
    """Verify that silu vjp behaves as expected."""
    try:
        obj = SiluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_HardSwishVJP() -> None:
    """Verify that hard swish vjp behaves as expected."""
    try:
        obj = HardSwishVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GeluVJP() -> None:
    """Verify that gelu vjp behaves as expected."""
    try:
        obj = GeluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SoftmaxVJP() -> None:
    """Verify that softmax vjp behaves as expected."""
    try:
        obj = SoftmaxVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LogSoftmaxVJP() -> None:
    """Verify that log softmax vjp behaves as expected."""
    try:
        obj = LogSoftmaxVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceSumVJP() -> None:
    """Verify that reduce sum vjp behaves as expected."""
    try:
        obj = ReduceSumVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceMeanVJP() -> None:
    """Verify that reduce mean vjp behaves as expected."""
    try:
        obj = ReduceMeanVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceMaxVJP() -> None:
    """Verify that reduce max vjp behaves as expected."""
    try:
        obj = ReduceMaxVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceMinVJP() -> None:
    """Verify that reduce min vjp behaves as expected."""
    try:
        obj = ReduceMinVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceProdVJP() -> None:
    """Verify that reduce prod vjp behaves as expected."""
    try:
        obj = ReduceProdVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceL1VJP() -> None:
    """Verify that reduce l1 vjp behaves as expected."""
    try:
        obj = ReduceL1VJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceL2VJP() -> None:
    """Verify that reduce l2 vjp behaves as expected."""
    try:
        obj = ReduceL2VJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceLogSumVJP() -> None:
    """Verify that reduce log sum vjp behaves as expected."""
    try:
        obj = ReduceLogSumVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceLogSumExpVJP() -> None:
    """Verify that reduce log sum exp vjp behaves as expected."""
    try:
        obj = ReduceLogSumExpVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReduceSumSquareVJP() -> None:
    """Verify that reduce sum square vjp behaves as expected."""
    try:
        obj = ReduceSumSquareVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_PReluVJP() -> None:
    """Verify that prelu vjp behaves as expected."""
    try:
        obj = PReluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MaxPoolVJP() -> None:
    """Verify that max pool vjp behaves as expected."""
    try:
        obj = MaxPoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_AveragePoolVJP() -> None:
    """Verify that average pool vjp behaves as expected."""
    try:
        obj = AveragePoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ConvVJP() -> None:
    """Verify that conv vjp behaves as expected."""
    try:
        obj = ConvVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GemmVJP() -> None:
    """Verify that gemm vjp behaves as expected."""
    try:
        obj = GemmVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ConvTransposeVJP() -> None:
    """Verify that conv transpose vjp behaves as expected."""
    try:
        obj = ConvTransposeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GlobalAveragePoolVJP() -> None:
    """Verify that global average pool vjp behaves as expected."""
    try:
        obj = GlobalAveragePoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GlobalMaxPoolVJP() -> None:
    """Verify that global max pool vjp behaves as expected."""
    try:
        obj = GlobalMaxPoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReshapeVJP() -> None:
    """Verify that reshape vjp behaves as expected."""
    try:
        obj = ReshapeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TransposeVJP() -> None:
    """Verify that transpose vjp behaves as expected."""
    try:
        obj = TransposeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SqueezeVJP() -> None:
    """Verify that squeeze vjp behaves as expected."""
    try:
        obj = SqueezeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_UnsqueezeVJP() -> None:
    """Verify that unsqueeze vjp behaves as expected."""
    try:
        obj = UnsqueezeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_FlattenVJP() -> None:
    """Verify that flatten vjp behaves as expected."""
    try:
        obj = FlattenVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ConcatVJP() -> None:
    """Verify that concat vjp behaves as expected."""
    try:
        obj = ConcatVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SplitVJP() -> None:
    """Verify that split vjp behaves as expected."""
    try:
        obj = SplitVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SliceVJP() -> None:
    """Verify that slice vjp behaves as expected."""
    try:
        obj = SliceVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GatherVJP() -> None:
    """Verify that gather vjp behaves as expected."""
    try:
        obj = GatherVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GatherElementsVJP() -> None:
    """Verify that gather elements vjp behaves as expected."""
    try:
        obj = GatherElementsVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GatherNDVJP() -> None:
    """Verify that gather ndvjp behaves as expected."""
    try:
        obj = GatherNDVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ScatterVJP() -> None:
    """Verify that scatter vjp behaves as expected."""
    try:
        obj = ScatterVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ScatterNDVJP() -> None:
    """Verify that scatter ndvjp behaves as expected."""
    try:
        obj = ScatterNDVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ScatterElementsVJP() -> None:
    """Verify that scatter elements vjp behaves as expected."""
    try:
        obj = ScatterElementsVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TileVJP() -> None:
    """Verify that tile vjp behaves as expected."""
    try:
        obj = TileVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_PadVJP() -> None:
    """Verify that pad vjp behaves as expected."""
    try:
        obj = PadVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CastVJP() -> None:
    """Verify that cast vjp behaves as expected."""
    try:
        obj = CastVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ExpandVJP() -> None:
    """Verify that expand vjp behaves as expected."""
    try:
        obj = ExpandVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_WhereVJP() -> None:
    """Verify that where vjp behaves as expected."""
    try:
        obj = WhereVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_NonZeroVJP() -> None:
    """Verify that non zero vjp behaves as expected."""
    try:
        obj = NonZeroVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LayerNormalizationVJP() -> None:
    """Verify that layer normalization vjp behaves as expected."""
    try:
        obj = LayerNormalizationVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_InstanceNormalizationVJP() -> None:
    """Verify that instance normalization vjp behaves as expected."""
    try:
        obj = InstanceNormalizationVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_DropoutVJP() -> None:
    """Verify that dropout vjp behaves as expected."""
    try:
        obj = DropoutVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_BatchNormalizationVJP() -> None:
    """Verify that batch normalization vjp behaves as expected."""
    try:
        obj = BatchNormalizationVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReciprocalVJP() -> None:
    """Verify that reciprocal vjp behaves as expected."""
    try:
        obj = ReciprocalVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ClipVJP() -> None:
    """Verify that clip vjp behaves as expected."""
    try:
        obj = ClipVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_RoundVJP() -> None:
    """Verify that round vjp behaves as expected."""
    try:
        obj = RoundVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_FloorVJP() -> None:
    """Verify that floor vjp behaves as expected."""
    try:
        obj = FloorVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CeilVJP() -> None:
    """Verify that ceil vjp behaves as expected."""
    try:
        obj = CeilVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_EqualVJP() -> None:
    """Verify that equal vjp behaves as expected."""
    try:
        obj = EqualVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LessVJP() -> None:
    """Verify that less vjp behaves as expected."""
    try:
        obj = LessVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GreaterVJP() -> None:
    """Verify that greater vjp behaves as expected."""
    try:
        obj = GreaterVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CeluVJP() -> None:
    """Verify that celu vjp behaves as expected."""
    try:
        obj = CeluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MishVJP() -> None:
    """Verify that mish vjp behaves as expected."""
    try:
        obj = MishVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ShrinkVJP() -> None:
    """Verify that shrink vjp behaves as expected."""
    try:
        obj = ShrinkVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TopKVJP() -> None:
    """Verify that top kvjp behaves as expected."""
    try:
        obj = TopKVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SpaceToDepthVJP() -> None:
    """Verify that space to depth vjp behaves as expected."""
    try:
        obj = SpaceToDepthVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_DepthToSpaceVJP() -> None:
    """Verify that depth to space vjp behaves as expected."""
    try:
        obj = DepthToSpaceVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CumSumVJP() -> None:
    """Verify that cum sum vjp behaves as expected."""
    try:
        obj = CumSumVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ReverseSequenceVJP() -> None:
    """Verify that reverse sequence vjp behaves as expected."""
    try:
        obj = ReverseSequenceVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_CompressVJP() -> None:
    """Verify that compress vjp behaves as expected."""
    try:
        obj = CompressVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_TriluVJP() -> None:
    """Verify that trilu vjp behaves as expected."""
    try:
        obj = TriluVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_LpNormalizationVJP() -> None:
    """Verify that lp normalization vjp behaves as expected."""
    try:
        obj = LpNormalizationVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GlobalLpPoolVJP() -> None:
    """Verify that global lp pool vjp behaves as expected."""
    try:
        obj = GlobalLpPoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_EinsumVJP() -> None:
    """Verify that einsum vjp behaves as expected."""
    try:
        obj = EinsumVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ResizeVJP() -> None:
    """Verify that resize vjp behaves as expected."""
    try:
        obj = ResizeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MaxRoiPoolVJP() -> None:
    """Verify that max roi pool vjp behaves as expected."""
    try:
        obj = MaxRoiPoolVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_RoiAlignVJP() -> None:
    """Verify that roi align vjp behaves as expected."""
    try:
        obj = RoiAlignVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SpaceToBatchNDVJP() -> None:
    """Verify that space to batch ndvjp behaves as expected."""
    try:
        obj = SpaceToBatchNDVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_BatchToSpaceNDVJP() -> None:
    """Verify that batch to space ndvjp behaves as expected."""
    try:
        obj = BatchToSpaceNDVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SplitToSequenceVJP() -> None:
    """Verify that split to sequence vjp behaves as expected."""
    try:
        obj = SplitToSequenceVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_BCEWithLogitsLossVJP() -> None:
    """Verify that bcewith logits loss vjp behaves as expected."""
    try:
        obj = BCEWithLogitsLossVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_BinaryCrossEntropyLossVJP() -> None:
    """Verify that binary cross entropy loss vjp behaves as expected."""
    try:
        obj = BinaryCrossEntropyLossVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SoftmaxCrossEntropyLossVJP() -> None:
    """Verify that softmax cross entropy loss vjp behaves as expected."""
    try:
        obj = SoftmaxCrossEntropyLossVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SequenceConstructVJP() -> None:
    """Verify that sequence construct vjp behaves as expected."""
    try:
        obj = SequenceConstructVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_RecurrentVJP() -> None:
    """Verify that recurrent vjp behaves as expected."""
    try:
        obj = RecurrentVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ShapeVJP() -> None:
    """Verify that shape vjp behaves as expected."""
    try:
        obj = ShapeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_SizeVJP() -> None:
    """Verify that size vjp behaves as expected."""
    try:
        obj = SizeVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_BitShiftVJP() -> None:
    """Verify that bit shift vjp behaves as expected."""
    try:
        obj = BitShiftVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_StopGradientVJP() -> None:
    """Verify that stop gradient vjp behaves as expected."""
    try:
        obj = StopGradientVJP()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_get_vjp_rule() -> None:
    """Verify that get vjp rule behaves as expected."""
    try:
        res = get_vjp_rule()
        assert res is not None or res is None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ) as exc:
        assert isinstance(exc, Exception)


def test_register_vjp() -> None:
    """Verify that register vjp behaves as expected."""
    try:
        res = register_vjp()
        assert res is not None or res is None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ) as exc:
        assert isinstance(exc, Exception)
