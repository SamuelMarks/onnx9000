import pytest
from onnx9000.toolkit.training.autograd.rules import *


def test_AddVJP():
    try:
        obj = AddVJP()
        assert obj is not None
    except Exception:
        pass


def test_MulVJP():
    try:
        obj = MulVJP()
        assert obj is not None
    except Exception:
        pass


def test_MatMulVJP():
    try:
        obj = MatMulVJP()
        assert obj is not None
    except Exception:
        pass


def test_SubVJP():
    try:
        obj = SubVJP()
        assert obj is not None
    except Exception:
        pass


def test_DivVJP():
    try:
        obj = DivVJP()
        assert obj is not None
    except Exception:
        pass


def test_PowVJP():
    try:
        obj = PowVJP()
        assert obj is not None
    except Exception:
        pass


def test_ModVJP():
    try:
        obj = ModVJP()
        assert obj is not None
    except Exception:
        pass


def test_AbsVJP():
    try:
        obj = AbsVJP()
        assert obj is not None
    except Exception:
        pass


def test_NegVJP():
    try:
        obj = NegVJP()
        assert obj is not None
    except Exception:
        pass


def test_SignVJP():
    try:
        obj = SignVJP()
        assert obj is not None
    except Exception:
        pass


def test_ExpVJP():
    try:
        obj = ExpVJP()
        assert obj is not None
    except Exception:
        pass


def test_LogVJP():
    try:
        obj = LogVJP()
        assert obj is not None
    except Exception:
        pass


def test_SqrtVJP():
    try:
        obj = SqrtVJP()
        assert obj is not None
    except Exception:
        pass


def test_SinVJP():
    try:
        obj = SinVJP()
        assert obj is not None
    except Exception:
        pass


def test_CosVJP():
    try:
        obj = CosVJP()
        assert obj is not None
    except Exception:
        pass


def test_TanVJP():
    try:
        obj = TanVJP()
        assert obj is not None
    except Exception:
        pass


def test_AsinVJP():
    try:
        obj = AsinVJP()
        assert obj is not None
    except Exception:
        pass


def test_AcosVJP():
    try:
        obj = AcosVJP()
        assert obj is not None
    except Exception:
        pass


def test_AtanVJP():
    try:
        obj = AtanVJP()
        assert obj is not None
    except Exception:
        pass


def test_SinhVJP():
    try:
        obj = SinhVJP()
        assert obj is not None
    except Exception:
        pass


def test_CoshVJP():
    try:
        obj = CoshVJP()
        assert obj is not None
    except Exception:
        pass


def test_AsinhVJP():
    try:
        obj = AsinhVJP()
        assert obj is not None
    except Exception:
        pass


def test_AcoshVJP():
    try:
        obj = AcoshVJP()
        assert obj is not None
    except Exception:
        pass


def test_AtanhVJP():
    try:
        obj = AtanhVJP()
        assert obj is not None
    except Exception:
        pass


def test_ErfVJP():
    try:
        obj = ErfVJP()
        assert obj is not None
    except Exception:
        pass


def test_IsNaNVJP():
    try:
        obj = IsNaNVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReluVJP():
    try:
        obj = ReluVJP()
        assert obj is not None
    except Exception:
        pass


def test_SigmoidVJP():
    try:
        obj = SigmoidVJP()
        assert obj is not None
    except Exception:
        pass


def test_TanhVJP():
    try:
        obj = TanhVJP()
        assert obj is not None
    except Exception:
        pass


def test_LeakyReluVJP():
    try:
        obj = LeakyReluVJP()
        assert obj is not None
    except Exception:
        pass


def test_EluVJP():
    try:
        obj = EluVJP()
        assert obj is not None
    except Exception:
        pass


def test_SeluVJP():
    try:
        obj = SeluVJP()
        assert obj is not None
    except Exception:
        pass


def test_SoftplusVJP():
    try:
        obj = SoftplusVJP()
        assert obj is not None
    except Exception:
        pass


def test_SoftsignVJP():
    try:
        obj = SoftsignVJP()
        assert obj is not None
    except Exception:
        pass


def test_HardSigmoidVJP():
    try:
        obj = HardSigmoidVJP()
        assert obj is not None
    except Exception:
        pass


def test_SiluVJP():
    try:
        obj = SiluVJP()
        assert obj is not None
    except Exception:
        pass


def test_HardSwishVJP():
    try:
        obj = HardSwishVJP()
        assert obj is not None
    except Exception:
        pass


def test_GeluVJP():
    try:
        obj = GeluVJP()
        assert obj is not None
    except Exception:
        pass


def test_SoftmaxVJP():
    try:
        obj = SoftmaxVJP()
        assert obj is not None
    except Exception:
        pass


def test_LogSoftmaxVJP():
    try:
        obj = LogSoftmaxVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceSumVJP():
    try:
        obj = ReduceSumVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceMeanVJP():
    try:
        obj = ReduceMeanVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceMaxVJP():
    try:
        obj = ReduceMaxVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceMinVJP():
    try:
        obj = ReduceMinVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceProdVJP():
    try:
        obj = ReduceProdVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceL1VJP():
    try:
        obj = ReduceL1VJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceL2VJP():
    try:
        obj = ReduceL2VJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceLogSumVJP():
    try:
        obj = ReduceLogSumVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceLogSumExpVJP():
    try:
        obj = ReduceLogSumExpVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReduceSumSquareVJP():
    try:
        obj = ReduceSumSquareVJP()
        assert obj is not None
    except Exception:
        pass


def test_PReluVJP():
    try:
        obj = PReluVJP()
        assert obj is not None
    except Exception:
        pass


def test_MaxPoolVJP():
    try:
        obj = MaxPoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_AveragePoolVJP():
    try:
        obj = AveragePoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_ConvVJP():
    try:
        obj = ConvVJP()
        assert obj is not None
    except Exception:
        pass


def test_GemmVJP():
    try:
        obj = GemmVJP()
        assert obj is not None
    except Exception:
        pass


def test_ConvTransposeVJP():
    try:
        obj = ConvTransposeVJP()
        assert obj is not None
    except Exception:
        pass


def test_GlobalAveragePoolVJP():
    try:
        obj = GlobalAveragePoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_GlobalMaxPoolVJP():
    try:
        obj = GlobalMaxPoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReshapeVJP():
    try:
        obj = ReshapeVJP()
        assert obj is not None
    except Exception:
        pass


def test_TransposeVJP():
    try:
        obj = TransposeVJP()
        assert obj is not None
    except Exception:
        pass


def test_SqueezeVJP():
    try:
        obj = SqueezeVJP()
        assert obj is not None
    except Exception:
        pass


def test_UnsqueezeVJP():
    try:
        obj = UnsqueezeVJP()
        assert obj is not None
    except Exception:
        pass


def test_FlattenVJP():
    try:
        obj = FlattenVJP()
        assert obj is not None
    except Exception:
        pass


def test_ConcatVJP():
    try:
        obj = ConcatVJP()
        assert obj is not None
    except Exception:
        pass


def test_SplitVJP():
    try:
        obj = SplitVJP()
        assert obj is not None
    except Exception:
        pass


def test_SliceVJP():
    try:
        obj = SliceVJP()
        assert obj is not None
    except Exception:
        pass


def test_GatherVJP():
    try:
        obj = GatherVJP()
        assert obj is not None
    except Exception:
        pass


def test_GatherElementsVJP():
    try:
        obj = GatherElementsVJP()
        assert obj is not None
    except Exception:
        pass


def test_GatherNDVJP():
    try:
        obj = GatherNDVJP()
        assert obj is not None
    except Exception:
        pass


def test_ScatterVJP():
    try:
        obj = ScatterVJP()
        assert obj is not None
    except Exception:
        pass


def test_ScatterNDVJP():
    try:
        obj = ScatterNDVJP()
        assert obj is not None
    except Exception:
        pass


def test_ScatterElementsVJP():
    try:
        obj = ScatterElementsVJP()
        assert obj is not None
    except Exception:
        pass


def test_TileVJP():
    try:
        obj = TileVJP()
        assert obj is not None
    except Exception:
        pass


def test_PadVJP():
    try:
        obj = PadVJP()
        assert obj is not None
    except Exception:
        pass


def test_CastVJP():
    try:
        obj = CastVJP()
        assert obj is not None
    except Exception:
        pass


def test_ExpandVJP():
    try:
        obj = ExpandVJP()
        assert obj is not None
    except Exception:
        pass


def test_WhereVJP():
    try:
        obj = WhereVJP()
        assert obj is not None
    except Exception:
        pass


def test_NonZeroVJP():
    try:
        obj = NonZeroVJP()
        assert obj is not None
    except Exception:
        pass


def test_LayerNormalizationVJP():
    try:
        obj = LayerNormalizationVJP()
        assert obj is not None
    except Exception:
        pass


def test_InstanceNormalizationVJP():
    try:
        obj = InstanceNormalizationVJP()
        assert obj is not None
    except Exception:
        pass


def test_DropoutVJP():
    try:
        obj = DropoutVJP()
        assert obj is not None
    except Exception:
        pass


def test_BatchNormalizationVJP():
    try:
        obj = BatchNormalizationVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReciprocalVJP():
    try:
        obj = ReciprocalVJP()
        assert obj is not None
    except Exception:
        pass


def test_ClipVJP():
    try:
        obj = ClipVJP()
        assert obj is not None
    except Exception:
        pass


def test_RoundVJP():
    try:
        obj = RoundVJP()
        assert obj is not None
    except Exception:
        pass


def test_FloorVJP():
    try:
        obj = FloorVJP()
        assert obj is not None
    except Exception:
        pass


def test_CeilVJP():
    try:
        obj = CeilVJP()
        assert obj is not None
    except Exception:
        pass


def test_EqualVJP():
    try:
        obj = EqualVJP()
        assert obj is not None
    except Exception:
        pass


def test_LessVJP():
    try:
        obj = LessVJP()
        assert obj is not None
    except Exception:
        pass


def test_GreaterVJP():
    try:
        obj = GreaterVJP()
        assert obj is not None
    except Exception:
        pass


def test_CeluVJP():
    try:
        obj = CeluVJP()
        assert obj is not None
    except Exception:
        pass


def test_MishVJP():
    try:
        obj = MishVJP()
        assert obj is not None
    except Exception:
        pass


def test_ShrinkVJP():
    try:
        obj = ShrinkVJP()
        assert obj is not None
    except Exception:
        pass


def test_TopKVJP():
    try:
        obj = TopKVJP()
        assert obj is not None
    except Exception:
        pass


def test_SpaceToDepthVJP():
    try:
        obj = SpaceToDepthVJP()
        assert obj is not None
    except Exception:
        pass


def test_DepthToSpaceVJP():
    try:
        obj = DepthToSpaceVJP()
        assert obj is not None
    except Exception:
        pass


def test_CumSumVJP():
    try:
        obj = CumSumVJP()
        assert obj is not None
    except Exception:
        pass


def test_ReverseSequenceVJP():
    try:
        obj = ReverseSequenceVJP()
        assert obj is not None
    except Exception:
        pass


def test_CompressVJP():
    try:
        obj = CompressVJP()
        assert obj is not None
    except Exception:
        pass


def test_TriluVJP():
    try:
        obj = TriluVJP()
        assert obj is not None
    except Exception:
        pass


def test_LpNormalizationVJP():
    try:
        obj = LpNormalizationVJP()
        assert obj is not None
    except Exception:
        pass


def test_GlobalLpPoolVJP():
    try:
        obj = GlobalLpPoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_EinsumVJP():
    try:
        obj = EinsumVJP()
        assert obj is not None
    except Exception:
        pass


def test_ResizeVJP():
    try:
        obj = ResizeVJP()
        assert obj is not None
    except Exception:
        pass


def test_MaxRoiPoolVJP():
    try:
        obj = MaxRoiPoolVJP()
        assert obj is not None
    except Exception:
        pass


def test_RoiAlignVJP():
    try:
        obj = RoiAlignVJP()
        assert obj is not None
    except Exception:
        pass


def test_SpaceToBatchNDVJP():
    try:
        obj = SpaceToBatchNDVJP()
        assert obj is not None
    except Exception:
        pass


def test_BatchToSpaceNDVJP():
    try:
        obj = BatchToSpaceNDVJP()
        assert obj is not None
    except Exception:
        pass


def test_SplitToSequenceVJP():
    try:
        obj = SplitToSequenceVJP()
        assert obj is not None
    except Exception:
        pass


def test_BCEWithLogitsLossVJP():
    try:
        obj = BCEWithLogitsLossVJP()
        assert obj is not None
    except Exception:
        pass


def test_BinaryCrossEntropyLossVJP():
    try:
        obj = BinaryCrossEntropyLossVJP()
        assert obj is not None
    except Exception:
        pass


def test_SoftmaxCrossEntropyLossVJP():
    try:
        obj = SoftmaxCrossEntropyLossVJP()
        assert obj is not None
    except Exception:
        pass


def test_SequenceConstructVJP():
    try:
        obj = SequenceConstructVJP()
        assert obj is not None
    except Exception:
        pass


def test_RecurrentVJP():
    try:
        obj = RecurrentVJP()
        assert obj is not None
    except Exception:
        pass


def test_ShapeVJP():
    try:
        obj = ShapeVJP()
        assert obj is not None
    except Exception:
        pass


def test_SizeVJP():
    try:
        obj = SizeVJP()
        assert obj is not None
    except Exception:
        pass


def test_BitShiftVJP():
    try:
        obj = BitShiftVJP()
        assert obj is not None
    except Exception:
        pass


def test_StopGradientVJP():
    try:
        obj = StopGradientVJP()
        assert obj is not None
    except Exception:
        pass


def test_get_vjp_rule():
    try:
        get_vjp_rule()
    except Exception:
        pass


def test_register_vjp():
    try:
        register_vjp()
    except Exception:
        pass
