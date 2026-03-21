import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, implementDynamicBatching, TensorType } from '../src/index.js';

describe('Dynamic Batching pass', () => {
  it('Replaces static batch size 1 with dynamic B for block inputs', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1, 224, 224, 3]));
    block.inputs.push(x);

    implementDynamicBatching(block);

    expect((block.inputs[0]?.type as TensorType).shape[0]).toBe('B');
  });

  it('Replaces static batch size 1 with dynamic B for operation outputs', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1, 224, 224, 3]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1, 224, 224, 3]));
    builder.addOp('relu', { x }, [out]);

    implementDynamicBatching(block);

    const op = block.operations[0];
    expect((op?.outputs[0]?.type as TensorType).shape[0]).toBe('B');
  });

  it('Ignores dimensions that are not 1 or not index 0', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [4, 1, 224, 3]));
    block.inputs.push(x);

    implementDynamicBatching(block);

    expect((block.inputs[0]?.type as TensorType).shape[0]).toBe(4);
    expect((block.inputs[0]?.type as TensorType).shape[1]).toBe(1);
  });
});
