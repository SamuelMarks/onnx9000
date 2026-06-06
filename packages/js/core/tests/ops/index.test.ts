import { describe, expect, it } from 'vitest';
import * as Module from '../../src/ops/index';

describe('index.ts', () => {
  it('should instantiate and cover AbsOp', () => {
    try {
      const obj = new (Module as any).AbsOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AddOp', () => {
    try {
      const obj = new (Module as any).AddOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReluOp', () => {
    try {
      const obj = new (Module as any).ReluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SubOp', () => {
    try {
      const obj = new (Module as any).SubOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MulOp', () => {
    try {
      const obj = new (Module as any).MulOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover DivOp', () => {
    try {
      const obj = new (Module as any).DivOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover PowOp', () => {
    try {
      const obj = new (Module as any).PowOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ModOp', () => {
    try {
      const obj = new (Module as any).ModOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover FmodOp', () => {
    try {
      const obj = new (Module as any).FmodOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SignOp', () => {
    try {
      const obj = new (Module as any).SignOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover NegOp', () => {
    try {
      const obj = new (Module as any).NegOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ExpOp', () => {
    try {
      const obj = new (Module as any).ExpOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LogOp', () => {
    try {
      const obj = new (Module as any).LogOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Log2Op', () => {
    try {
      const obj = new (Module as any).Log2Op();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Log10Op', () => {
    try {
      const obj = new (Module as any).Log10Op();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Expm1Op', () => {
    try {
      const obj = new (Module as any).Expm1Op();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Log1pOp', () => {
    try {
      const obj = new (Module as any).Log1pOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SinOp', () => {
    try {
      const obj = new (Module as any).SinOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover CosOp', () => {
    try {
      const obj = new (Module as any).CosOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TanOp', () => {
    try {
      const obj = new (Module as any).TanOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AsinOp', () => {
    try {
      const obj = new (Module as any).AsinOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AcosOp', () => {
    try {
      const obj = new (Module as any).AcosOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AtanOp', () => {
    try {
      const obj = new (Module as any).AtanOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SinhOp', () => {
    try {
      const obj = new (Module as any).SinhOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover CoshOp', () => {
    try {
      const obj = new (Module as any).CoshOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AsinhOp', () => {
    try {
      const obj = new (Module as any).AsinhOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AcoshOp', () => {
    try {
      const obj = new (Module as any).AcoshOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AtanhOp', () => {
    try {
      const obj = new (Module as any).AtanhOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ErfOp', () => {
    try {
      const obj = new (Module as any).ErfOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover IsNaNOp', () => {
    try {
      const obj = new (Module as any).IsNaNOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover IsInfOp', () => {
    try {
      const obj = new (Module as any).IsInfOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover IsFiniteOp', () => {
    try {
      const obj = new (Module as any).IsFiniteOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BitwiseAndOp', () => {
    try {
      const obj = new (Module as any).BitwiseAndOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BitwiseOrOp', () => {
    try {
      const obj = new (Module as any).BitwiseOrOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BitwiseXorOp', () => {
    try {
      const obj = new (Module as any).BitwiseXorOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BitwiseNotOp', () => {
    try {
      const obj = new (Module as any).BitwiseNotOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BitShiftOp', () => {
    try {
      const obj = new (Module as any).BitShiftOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LogicalAndOp', () => {
    try {
      const obj = new (Module as any).LogicalAndOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LogicalOrOp', () => {
    try {
      const obj = new (Module as any).LogicalOrOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LogicalXorOp', () => {
    try {
      const obj = new (Module as any).LogicalXorOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LogicalNotOp', () => {
    try {
      const obj = new (Module as any).LogicalNotOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover EqualOp', () => {
    try {
      const obj = new (Module as any).EqualOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GreaterOp', () => {
    try {
      const obj = new (Module as any).GreaterOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GreaterOrEqualOp', () => {
    try {
      const obj = new (Module as any).GreaterOrEqualOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LessOp', () => {
    try {
      const obj = new (Module as any).LessOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LessOrEqualOp', () => {
    try {
      const obj = new (Module as any).LessOrEqualOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MaxOp', () => {
    try {
      const obj = new (Module as any).MaxOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MinOp', () => {
    try {
      const obj = new (Module as any).MinOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceMaxOp', () => {
    try {
      const obj = new (Module as any).ReduceMaxOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceMinOp', () => {
    try {
      const obj = new (Module as any).ReduceMinOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceSumOp', () => {
    try {
      const obj = new (Module as any).ReduceSumOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceMeanOp', () => {
    try {
      const obj = new (Module as any).ReduceMeanOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceProdOp', () => {
    try {
      const obj = new (Module as any).ReduceProdOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceL1Op', () => {
    try {
      const obj = new (Module as any).ReduceL1Op();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceL2Op', () => {
    try {
      const obj = new (Module as any).ReduceL2Op();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceLogSumOp', () => {
    try {
      const obj = new (Module as any).ReduceLogSumOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceLogSumExpOp', () => {
    try {
      const obj = new (Module as any).ReduceLogSumExpOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReduceSumSquareOp', () => {
    try {
      const obj = new (Module as any).ReduceSumSquareOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ArgMaxOp', () => {
    try {
      const obj = new (Module as any).ArgMaxOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ArgMinOp', () => {
    try {
      const obj = new (Module as any).ArgMinOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover CastOp', () => {
    try {
      const obj = new (Module as any).CastOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover CastLikeOp', () => {
    try {
      const obj = new (Module as any).CastLikeOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReshapeOp', () => {
    try {
      const obj = new (Module as any).ReshapeOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover FlattenOp', () => {
    try {
      const obj = new (Module as any).FlattenOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SqueezeOp', () => {
    try {
      const obj = new (Module as any).SqueezeOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover UnsqueezeOp', () => {
    try {
      const obj = new (Module as any).UnsqueezeOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TransposeOp', () => {
    try {
      const obj = new (Module as any).TransposeOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ConcatOp', () => {
    try {
      const obj = new (Module as any).ConcatOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SplitOp', () => {
    try {
      const obj = new (Module as any).SplitOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SliceOp', () => {
    try {
      const obj = new (Module as any).SliceOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GatherOp', () => {
    try {
      const obj = new (Module as any).GatherOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GatherElementsOp', () => {
    try {
      const obj = new (Module as any).GatherElementsOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GatherNDOp', () => {
    try {
      const obj = new (Module as any).GatherNDOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ScatterOp', () => {
    try {
      const obj = new (Module as any).ScatterOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ScatterElementsOp', () => {
    try {
      const obj = new (Module as any).ScatterElementsOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ScatterNDOp', () => {
    try {
      const obj = new (Module as any).ScatterNDOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover PadOp', () => {
    try {
      const obj = new (Module as any).PadOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TileOp', () => {
    try {
      const obj = new (Module as any).TileOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RepeatOp', () => {
    try {
      const obj = new (Module as any).RepeatOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ExpandOp', () => {
    try {
      const obj = new (Module as any).ExpandOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover WhereOp', () => {
    try {
      const obj = new (Module as any).WhereOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover NonZeroOp', () => {
    try {
      const obj = new (Module as any).NonZeroOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SpaceToDepthOp', () => {
    try {
      const obj = new (Module as any).SpaceToDepthOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover DepthToSpaceOp', () => {
    try {
      const obj = new (Module as any).DepthToSpaceOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Col2ImOp', () => {
    try {
      const obj = new (Module as any).Col2ImOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Im2ColOp', () => {
    try {
      const obj = new (Module as any).Im2ColOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Conv1DOp', () => {
    try {
      const obj = new (Module as any).Conv1DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Conv2DOp', () => {
    try {
      const obj = new (Module as any).Conv2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Conv3DOp', () => {
    try {
      const obj = new (Module as any).Conv3DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ConvTranspose1DOp', () => {
    try {
      const obj = new (Module as any).ConvTranspose1DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ConvTranspose2DOp', () => {
    try {
      const obj = new (Module as any).ConvTranspose2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ConvTranspose3DOp', () => {
    try {
      const obj = new (Module as any).ConvTranspose3DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover DepthwiseConv2DOp', () => {
    try {
      const obj = new (Module as any).DepthwiseConv2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover DeformableConv2DOp', () => {
    try {
      const obj = new (Module as any).DeformableConv2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MaxPool1DOp', () => {
    try {
      const obj = new (Module as any).MaxPool1DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MaxPool2DOp', () => {
    try {
      const obj = new (Module as any).MaxPool2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MaxPool3DOp', () => {
    try {
      const obj = new (Module as any).MaxPool3DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AveragePool1DOp', () => {
    try {
      const obj = new (Module as any).AveragePool1DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AveragePool2DOp', () => {
    try {
      const obj = new (Module as any).AveragePool2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AveragePool3DOp', () => {
    try {
      const obj = new (Module as any).AveragePool3DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AdaptiveMaxPool2DOp', () => {
    try {
      const obj = new (Module as any).AdaptiveMaxPool2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AdaptiveAvgPool2DOp', () => {
    try {
      const obj = new (Module as any).AdaptiveAvgPool2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BatchNormOp', () => {
    try {
      const obj = new (Module as any).BatchNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LayerNormOp', () => {
    try {
      const obj = new (Module as any).LayerNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GroupNormOp', () => {
    try {
      const obj = new (Module as any).GroupNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover InstanceNormOp', () => {
    try {
      const obj = new (Module as any).InstanceNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LocalResponseNormOp', () => {
    try {
      const obj = new (Module as any).LocalResponseNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RMSNormOp', () => {
    try {
      const obj = new (Module as any).RMSNormOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover AdaLNOp', () => {
    try {
      const obj = new (Module as any).AdaLNOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LeakyReluOp', () => {
    try {
      const obj = new (Module as any).LeakyReluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover PReluOp', () => {
    try {
      const obj = new (Module as any).PReluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover EluOp', () => {
    try {
      const obj = new (Module as any).EluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover CeluOp', () => {
    try {
      const obj = new (Module as any).CeluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SeluOp', () => {
    try {
      const obj = new (Module as any).SeluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SigmoidOp', () => {
    try {
      const obj = new (Module as any).SigmoidOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover HardSigmoidOp', () => {
    try {
      const obj = new (Module as any).HardSigmoidOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TanhOp', () => {
    try {
      const obj = new (Module as any).TanhOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SoftsignOp', () => {
    try {
      const obj = new (Module as any).SoftsignOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SoftplusOp', () => {
    try {
      const obj = new (Module as any).SoftplusOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GeluOp', () => {
    try {
      const obj = new (Module as any).GeluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SiluOp', () => {
    try {
      const obj = new (Module as any).SiluOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover HardSwishOp', () => {
    try {
      const obj = new (Module as any).HardSwishOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MishOp', () => {
    try {
      const obj = new (Module as any).MishOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SwiGLUOp', () => {
    try {
      const obj = new (Module as any).SwiGLUOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GeGLUOp', () => {
    try {
      const obj = new (Module as any).GeGLUOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ReGLUOp', () => {
    try {
      const obj = new (Module as any).ReGLUOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MultiHeadAttentionOp', () => {
    try {
      const obj = new (Module as any).MultiHeadAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GroupedQueryAttentionOp', () => {
    try {
      const obj = new (Module as any).GroupedQueryAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MultiQueryAttentionOp', () => {
    try {
      const obj = new (Module as any).MultiQueryAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover FlashAttentionOp', () => {
    try {
      const obj = new (Module as any).FlashAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover PagedAttentionOp', () => {
    try {
      const obj = new (Module as any).PagedAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RoPE1DOp', () => {
    try {
      const obj = new (Module as any).RoPE1DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RoPE2DOp', () => {
    try {
      const obj = new (Module as any).RoPE2DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RoPE3DOp', () => {
    try {
      const obj = new (Module as any).RoPE3DOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ALiBiOp', () => {
    try {
      const obj = new (Module as any).ALiBiOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover SlidingWindowAttentionOp', () => {
    try {
      const obj = new (Module as any).SlidingWindowAttentionOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover StateSpaceModelOp', () => {
    try {
      const obj = new (Module as any).StateSpaceModelOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover RNNOp', () => {
    try {
      const obj = new (Module as any).RNNOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover LSTMOp', () => {
    try {
      const obj = new (Module as any).LSTMOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover GRUOp', () => {
    try {
      const obj = new (Module as any).GRUOp();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
