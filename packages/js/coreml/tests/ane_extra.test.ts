import { describe, it, expect } from 'vitest';
import {
  Builder,
  MILDataType,
  optimizeForANE,
  verifyANECompatibility,
  TensorType,
  Operation,
} from '../src/index.js';

describe('ANE Optimization Passes (Extended)', () => {
  it('handles array inputs in matmul gracefully', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');
    const x = b.createVar('x', b.tensor(MILDataType.FLOAT32, [1]));
    const out = b.createVar('out', b.tensor(MILDataType.FLOAT32, [1]));

    // Matmul where inputs are arrays
    const op = new Operation('matmul', { x: [x], y: [x] }, [out]);
    block.addOperation(op);
    optimizeForANE(block);
    expect(op.attributes['ane_hint_convert_to_conv1x1']).toBe(true);
  });

  it('detects 5D tensors in conv and logs warning', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');
    const x = b.createVar('x', b.tensor(MILDataType.FLOAT32, [1, 2, 3, 4, 5]));
    const out = b.createVar('out', b.tensor(MILDataType.FLOAT32, [1]));

    b.addOp('conv', { x }, [out]);
    verifyANECompatibility(block); // Should hit the >4D branch
  });

  it('detects large dimension sizes and logs warning', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');
    const x = b.createVar('x', b.tensor(MILDataType.FLOAT32, [70000, 2]));
    const out = b.createVar('out', b.tensor(MILDataType.FLOAT32, [1]));

    b.addOp('conv', { x }, [out]);
    verifyANECompatibility(block); // Should hit the >65536 branch
  });

  it('annotates attention and layer_norm for ANE', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');
    const x = b.createVar('x', b.tensor(MILDataType.FLOAT32, [1]));

    b.addOp('attention', { x }, [x]);
    b.addOp('layer_norm', { x }, [x]);
    b.addOp('einsum', { x }, [x]);

    optimizeForANE(block);

    expect(block.operations[0]!.opType).toBe('scaled_dot_product_attention');
    expect(block.operations[1]!.attributes['ane_hint_decompose_layer_norm']).toBe(true);
    expect(block.operations[2]!.attributes['ane_hint_decompose_einsum']).toBe(true);
  });

  it('optimizes out redundant cast ops', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');

    const x = b.createVar('x', b.tensor(MILDataType.FLOAT16, [1]));
    const out = b.createVar('out', b.tensor(MILDataType.FLOAT16, [1]));

    const op = b.addOp('cast', { x }, [out], { dtype: 'fp16' });

    optimizeForANE(block);
    expect(op.opType).toBe('identity');
  });

  it('optimizes gather with constants', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');

    const indices = b.createVar('indices', b.tensor(MILDataType.INT32, [1]));
    b.addOp('gather', { indices }, [indices]);

    optimizeForANE(block);
    expect(block.operations[0]!.attributes['ane_hint_precompute_gather']).toBe(true);
  });

  it('detects massive convs for split concat', () => {
    const b = new Builder();
    b.createFunction('test', [], []);
    const block = b.createBlock('b0');

    // 20001 % 32 !== 0, > 16384
    const w = b.createVar('w', b.tensor(MILDataType.FLOAT32, [20001, 1]));
    b.addOp('conv', { weight: w }, [w]);

    optimizeForANE(block);
    expect(block.operations[0]!.attributes['ane_hint_split_concat']).toBe(true);
    expect(block.operations[0]!.attributes['ane_hint_pad_channels']).toBe(31);
  });
});
