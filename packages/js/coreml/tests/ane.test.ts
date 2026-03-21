import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, optimizeForANE, verifyANECompatibility } from '../src/index.js';

describe('ANE Optimization Passes', () => {
  it('Forces CAST inputs to FP16', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1]));

    builder.addOp('cast', { x }, [out], { dtype: 'fp32' });

    optimizeForANE(block);

    const op = block.operations[0];
    expect(op?.attributes['dtype']).toBe('fp16');
    expect(op?.outputs[0]?.type.toString()).toContain('fp16');
  });

  it('Annotates MatMul for 1x1 Conv conversion', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const y = builder.createVar('y', builder.tensor(MILDataType.FLOAT32, [1]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1]));

    builder.addOp('matmul', { x, y }, [out]);

    optimizeForANE(block);

    const op = block.operations[0];
    expect(op?.attributes['ane_hint_convert_to_conv1x1']).toBe(true);
  });

  it('Replaces Swish with HardSwish', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1]));

    builder.addOp('swish', { x }, [out]);

    optimizeForANE(block);

    const op = block.operations[0];
    expect(op?.opType).toBe('hard_swish');
  });
});
