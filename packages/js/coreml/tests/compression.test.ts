import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, applyCompression } from '../src/index.js';

describe('Compression Optimization Passes', () => {
  it('Applies INT8 Weight Quantization (W8A16)', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // Create a 1000-element fp32 tensor (4000 bytes)
    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT32, [1000]));
    builder.addOp('const', {}, [x], { value: new Float32Array(1000) });

    const report = applyCompression(block, { mode: 'w8a16', groupSize: 32, reportReduction: true });

    const op = block.operations[0];
    expect(op?.opType).toBe('constexpr_affine_dequantize');
    expect(op?.attributes['quant_type']).toBe('int8');
    expect(op?.attributes['group_size']).toBe(32);

    // FP32 -> INT8 should be 75% reduction
    expect(report.reductionPercentage).toBe(75);
  });

  it('Applies INT4 Weight Quantization (W4A16)', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // Create a 1000-element fp32 tensor (4000 bytes)
    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT32, [1000]));
    builder.addOp('const', {}, [x], { value: new Float32Array(1000) });

    const report = applyCompression(block, { mode: 'w4a16', groupSize: 64, reportReduction: true });

    const op = block.operations[0];
    expect(op?.opType).toBe('constexpr_affine_dequantize');
    expect(op?.attributes['quant_type']).toBe('int4');
    expect(op?.attributes['group_size']).toBe(64);

    // FP32 -> INT4 should be 87.5% reduction
    expect(report.reductionPercentage).toBe(87.5);
  });

  it('Applies Palettization Compression', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT32, [1000]));
    builder.addOp('const', {}, [x], { value: new Float32Array(1000) });

    applyCompression(block, { mode: 'palettization' });

    const op = block.operations[0];
    expect(op?.opType).toBe('constexpr_lut_dequantize');
  });

  it('Applies Sparse Compression', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT32, [1000]));
    builder.addOp('const', {}, [x], { value: new Float32Array(1000) });

    applyCompression(block, { mode: 'sparse' });

    const op = block.operations[0];
    expect(op?.opType).toBe('constexpr_sparse_dequantize');
  });
});
