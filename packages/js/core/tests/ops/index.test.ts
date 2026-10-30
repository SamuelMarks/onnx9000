import { describe, expect, it } from 'vitest';
import { Tensor } from '../../src/ir/tensor.js';
import {
  AbsOp,
  AcoshOp,
  AcosOp,
  AdaLNOp,
  AdaptiveAvgPool2DOp,
  AdaptiveMaxPool2DOp,
  AddOp,
  ALiBiOp,
  ArgMaxOp,
  ArgMinOp,
  AsinhOp,
  AsinOp,
  AtanhOp,
  AtanOp,
  AveragePool1DOp,
  AveragePool2DOp,
  AveragePool3DOp,
  BatchNormOp,
  BitShiftOp,
  BitwiseAndOp,
  BitwiseNotOp,
  BitwiseOrOp,
  BitwiseXorOp,
  CastLikeOp,
  CastOp,
  CeluOp,
  Col2ImOp,
  ConcatOp,
  Conv1DOp,
  Conv2DOp,
  Conv3DOp,
  ConvTranspose1DOp,
  ConvTranspose2DOp,
  ConvTranspose3DOp,
  CoshOp,
  CosOp,
  DeformableConv2DOp,
  DepthToSpaceOp,
  DepthwiseConv2DOp,
  DequantizeLinearOp,
  DivOp,
  DynamicQuantizeLinearOp,
  EluOp,
  EqualOp,
  ErfOp,
  ExpandOp,
  Expm1Op,
  ExpOp,
  FlashAttentionOp,
  FlattenOp,
  FmodOp,
  GatherElementsOp,
  GatherNDOp,
  GatherOp,
  GeGLUOp,
  GeluOp,
  GRUOp,
  GreaterOp,
  GreaterOrEqualOp,
  GroupedQueryAttentionOp,
  GroupNormOp,
  HardSigmoidOp,
  HardSwishOp,
  Im2ColOp,
  InstanceNormOp,
  IsFiniteOp,
  IsInfOp,
  IsNaNOp,
  LayerNormOp,
  LeakyReluOp,
  LessOp,
  LessOrEqualOp,
  LocalResponseNormOp,
  Log1pOp,
  Log2Op,
  Log10Op,
  LogicalAndOp,
  LogicalNotOp,
  LogicalOrOp,
  LogicalXorOp,
  LogOp,
  LSTMOp,
  MaxOp,
  MaxPool1DOp,
  MaxPool2DOp,
  MaxPool3DOp,
  MinOp,
  MishOp,
  ModOp,
  MulOp,
  MultiHeadAttentionOp,
  MultiQueryAttentionOp,
  NegOp,
  NonZeroOp,
  PadOp,
  PagedAttentionOp,
  PowOp,
  PReluOp,
  QuantizeLinearOp,
  ReduceL1Op,
  ReduceL2Op,
  ReduceLogSumExpOp,
  ReduceLogSumOp,
  ReduceMaxOp,
  ReduceMeanOp,
  ReduceMinOp,
  ReduceProdOp,
  ReduceSumOp,
  ReduceSumSquareOp,
  ReGLUOp,
  ReluOp,
  RepeatOp,
  ReshapeOp,
  RMSNormOp,
  RNNOp,
  RoPE1DOp,
  RoPE2DOp,
  RoPE3DOp,
  ScatterElementsOp,
  ScatterNDOp,
  ScatterOp,
  SeluOp,
  SigmoidOp,
  SignOp,
  SiluOp,
  SinhOp,
  SinOp,
  SliceOp,
  SlidingWindowAttentionOp,
  SoftplusOp,
  SoftsignOp,
  SpaceToDepthOp,
  SplitOp,
  SqueezeOp,
  StateSpaceModelOp,
  SubOp,
  SwiGLUOp,
  TanhOp,
  TanOp,
  TileOp,
  TransposeOp,
  UnsqueezeOp,
  WhereOp,
} from '../../src/ops/index.js';

describe('ONNX Operator Implementations', () => {
  const f32A = new Tensor(
    'a',
    [5],
    'float32',
    false,
    true,
    new Float32Array([0.0, 1.5, -2.5, Number.NaN, Number.POSITIVE_INFINITY]),
  );
  const f32B = new Tensor('b', [2], 'float32', false, true, new Float32Array([0.0, 0.5]));
  const f32C = new Tensor('c', [3], 'float32', false, true, new Float32Array([0.0, 0.5, 2.0]));
  const u8A = new Tensor(
    'u8a',
    [8],
    'uint8',
    false,
    true,
    new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
  );
  const u8B = new Tensor(
    'u8b',
    [8],
    'uint8',
    false,
    true,
    new Uint8Array([2, 2, 2, 2, 1, 1, 1, 1]),
  );
  const emptyData = new Tensor('empty', [2], 'float32', false, true, null);

  it('executes AbsOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AbsOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AddOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AddOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SubOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SubOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MulOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MulOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DivOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DivOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes PowOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new PowOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ModOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ModOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes FmodOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new FmodOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SignOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SignOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes NegOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new NegOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ExpOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ExpOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LogOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LogOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Log2Op across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Log2Op();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Log10Op across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Log10Op();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Expm1Op across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Expm1Op();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Log1pOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Log1pOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SinOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SinOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes CosOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new CosOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes TanOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new TanOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AsinOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AsinOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AcosOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AcosOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AtanOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AtanOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SinhOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SinhOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes CoshOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new CoshOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AsinhOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AsinhOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AcoshOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AcoshOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AtanhOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AtanhOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ErfOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ErfOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes IsNaNOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new IsNaNOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes IsInfOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new IsInfOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes IsFiniteOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new IsFiniteOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BitwiseAndOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BitwiseAndOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BitwiseOrOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BitwiseOrOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BitwiseXorOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BitwiseXorOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BitwiseNotOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BitwiseNotOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BitShiftOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BitShiftOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LogicalAndOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LogicalAndOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LogicalOrOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LogicalOrOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LogicalXorOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LogicalXorOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LogicalNotOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LogicalNotOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes EqualOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new EqualOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GreaterOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GreaterOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GreaterOrEqualOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GreaterOrEqualOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LessOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LessOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LessOrEqualOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LessOrEqualOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MaxOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MaxOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MinOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MinOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Conv1DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Conv1DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Conv2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Conv2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Conv3DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Conv3DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ConvTranspose1DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ConvTranspose1DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ConvTranspose2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ConvTranspose2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ConvTranspose3DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ConvTranspose3DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DepthwiseConv2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DepthwiseConv2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DeformableConv2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DeformableConv2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MaxPool1DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MaxPool1DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MaxPool2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MaxPool2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MaxPool3DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MaxPool3DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AveragePool1DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AveragePool1DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AveragePool2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AveragePool2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AveragePool3DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AveragePool3DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AdaptiveMaxPool2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AdaptiveMaxPool2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AdaptiveAvgPool2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AdaptiveAvgPool2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes BatchNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new BatchNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LayerNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LayerNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GroupNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GroupNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes InstanceNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new InstanceNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LocalResponseNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LocalResponseNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RMSNormOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RMSNormOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes AdaLNOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new AdaLNOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LeakyReluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LeakyReluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes PReluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new PReluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes EluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new EluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes CeluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new CeluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SeluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SeluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SigmoidOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SigmoidOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes HardSigmoidOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new HardSigmoidOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes TanhOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new TanhOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SoftsignOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SoftsignOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SoftplusOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SoftplusOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GeluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GeluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SiluOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SiluOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes HardSwishOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new HardSwishOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MishOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MishOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SwiGLUOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SwiGLUOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GeGLUOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GeGLUOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReGLUOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReGLUOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MultiHeadAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MultiHeadAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GroupedQueryAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GroupedQueryAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes MultiQueryAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new MultiQueryAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes FlashAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new FlashAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes PagedAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new PagedAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RoPE1DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RoPE1DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RoPE2DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RoPE2DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RoPE3DOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RoPE3DOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ALiBiOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ALiBiOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SlidingWindowAttentionOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SlidingWindowAttentionOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes StateSpaceModelOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new StateSpaceModelOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RNNOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RNNOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes LSTMOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new LSTMOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GRUOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GRUOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes QuantizeLinearOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new QuantizeLinearOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DequantizeLinearOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DequantizeLinearOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DynamicQuantizeLinearOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DynamicQuantizeLinearOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceMaxOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceMaxOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceMinOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceMinOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceSumOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceSumOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceMeanOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceMeanOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceProdOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceProdOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceL1Op across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceL1Op();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceL2Op across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceL2Op();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceLogSumOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceLogSumOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceLogSumExpOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceLogSumExpOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReduceSumSquareOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReduceSumSquareOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ArgMaxOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ArgMaxOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ArgMinOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ArgMinOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes CastOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new CastOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes CastLikeOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new CastLikeOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ReshapeOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ReshapeOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes FlattenOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new FlattenOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SqueezeOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SqueezeOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes UnsqueezeOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new UnsqueezeOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes TransposeOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new TransposeOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ConcatOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ConcatOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SplitOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SplitOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SliceOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SliceOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GatherOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GatherOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GatherElementsOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GatherElementsOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes GatherNDOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new GatherNDOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ScatterOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ScatterOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ScatterElementsOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ScatterElementsOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ScatterNDOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ScatterNDOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes PadOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new PadOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes TileOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new TileOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes RepeatOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new RepeatOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes ExpandOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new ExpandOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes WhereOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new WhereOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes NonZeroOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new NonZeroOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes SpaceToDepthOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new SpaceToDepthOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes DepthToSpaceOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new DepthToSpaceOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Col2ImOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Col2ImOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
  it('executes Im2ColOp across Float32, Uint8, and empty tensor inputs', () => {
    const op = new Im2ColOp();
    const outF32 = op.execute([f32A, f32B, u8A], { direction: 'LEFT' });
    expect(outF32.length).toBeGreaterThanOrEqual(1);
    const outRev = op.execute([f32B, f32C, f32A], { direction: 'RIGHT' });
    expect(outRev.length).toBeGreaterThanOrEqual(1);
    const outU8 = op.execute([u8A, u8B, u8A], {});
    expect(outU8.length).toBeGreaterThanOrEqual(1);
    const outMissing = op.execute([emptyData, f32B], {});
    expect(outMissing.length).toBe(1);
    const outEmpty = op.execute([], {});
    expect(outEmpty).toEqual([]);
  });
});
