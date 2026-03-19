import { describe, it, expect } from 'vitest';
import { Type, Value, BlockArgument, Block, Region, Operation } from '../src/ir/core.js';
import * as tensor from '../src/dialects/web/tensor.js';
import * as mhlo from '../src/dialects/web/mhlo.js';
import { lowerONNXToMHLO } from '../src/passes/lower_onnx_to_mhlo.js';
import { Graph, ValueInfo, Node } from '@onnx9000/core';
import { TensorType } from '../src/dialects/web/tensor.js';

describe('IR Core', () => {
  it('should create regions, blocks, and operations', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    const type: Type = { id: 'f32' };
    const arg1 = block.addArgument(type);
    const arg2 = block.addArgument(type);

    const op = new Operation('custom.add', [arg1, arg2], [type]);
    block.pushOperation(op);

    expect(region.blocks.length).toBe(1);
    expect(block.args.length).toBe(2);
    expect(block.operations.length).toBe(1);
    expect(block.operations[0].opcode).toBe('custom.add');
    expect(block.operations[0].results.length).toBe(1);
  });
});

describe('Web Tensor Dialect', () => {
  it('should create tensor extraction', () => {
    const tensorType = new TensorType([10, 10], 'float32');
    const indexType: Type = { id: 'i32' };
    const resultType: Type = { id: 'float32' };

    const dummyVal1 = new Value(tensorType, {} as Operation);
    const dummyIdx1 = new Value(indexType, {} as Operation);

    const op = tensor.extract(dummyVal1, [dummyIdx1], resultType);
    expect(op.opcode).toBe('web.tensor.extract');
    expect(op.operands.length).toBe(2);
  });

  it('should create tensor insertion', () => {
    const tensorType = new TensorType([10, 10], 'float32');
    const indexType: Type = { id: 'i32' };

    const dummyVal1 = new Value(tensorType, {} as Operation);
    const dummyScalar = new Value({ id: 'float32' }, {} as Operation);
    const dummyIdx1 = new Value(indexType, {} as Operation);

    const op = tensor.insert(dummyVal1, dummyScalar, [dummyIdx1], tensorType);
    expect(op.opcode).toBe('web.tensor.insert');
    expect(op.operands.length).toBe(3);
  });

  it('should create tensor splat', () => {
    const dummyScalar = new Value({ id: 'float32' }, {} as Operation);
    const tensorType = new TensorType([10, 10], 'float32');
    const op = tensor.splat(dummyScalar, tensorType);
    expect(op.opcode).toBe('web.tensor.splat');
  });

  it('should create tensor pad', () => {
    const tensorType = new TensorType([10, 10], 'float32');
    const dummyVal1 = new Value(tensorType, {} as Operation);
    const dummyScalar = new Value({ id: 'float32' }, {} as Operation);
    const op = tensor.pad(dummyVal1, dummyScalar, [1, 1], [1, 1], [0, 0], tensorType);
    expect(op.opcode).toBe('web.tensor.pad');
    expect(op.attributes.edgePaddingLow).toEqual([1, 1]);
  });
});

describe('Web MHLO Dialect', () => {
  it('should create basic math ops', () => {
    const type = new TensorType([10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value(type, {} as Operation);

    expect(mhlo.add(val1, val2, type).opcode).toBe('web.mhlo.add');
    expect(mhlo.subtract(val1, val2, type).opcode).toBe('web.mhlo.subtract');
    expect(mhlo.multiply(val1, val2, type).opcode).toBe('web.mhlo.multiply');
    expect(mhlo.divide(val1, val2, type).opcode).toBe('web.mhlo.divide');
    expect(mhlo.maximum(val1, val2, type).opcode).toBe('web.mhlo.maximum');
    expect(mhlo.minimum(val1, val2, type).opcode).toBe('web.mhlo.minimum');
    expect(mhlo.exponential(val1, type).opcode).toBe('web.mhlo.exponential');
    expect(mhlo.log(val1, type).opcode).toBe('web.mhlo.log');
    expect(mhlo.cosine(val1, type).opcode).toBe('web.mhlo.cosine');
    expect(mhlo.sine(val1, type).opcode).toBe('web.mhlo.sine');
  });

  it('should create dot and conv', () => {
    const type = new TensorType([10, 10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value(type, {} as Operation);

    expect(mhlo.dot(val1, val2, type).opcode).toBe('web.mhlo.dot');
    const conv = mhlo.convolution(
      val1,
      val2,
      [1, 1],
      [
        [0, 0],
        [0, 0],
      ],
      [1, 1],
      [1, 1],
      [false, false],
      type,
    );
    expect(conv.opcode).toBe('web.mhlo.convolution');
    expect(conv.attributes.windowStrides).toEqual([1, 1]);
  });

  it('should create structural ops', () => {
    const type = new TensorType([10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value(type, {} as Operation);

    expect(mhlo.select(val1, val2, val2, type).opcode).toBe('web.mhlo.select');
    expect(mhlo.broadcastInDim(val1, [0], type).opcode).toBe('web.mhlo.broadcast_in_dim');
    expect(mhlo.reshape(val1, type).opcode).toBe('web.mhlo.reshape');
    expect(mhlo.transpose(val1, [1, 0], type).opcode).toBe('web.mhlo.transpose');
    expect(mhlo.concatenate([val1, val2], 0, type).opcode).toBe('web.mhlo.concatenate');
    expect(mhlo.slice(val1, [0], [5], [1], type).opcode).toBe('web.mhlo.slice');
    expect(mhlo.dynamicSlice(val1, [val2], [5], type).opcode).toBe('web.mhlo.dynamic_slice');
    expect(mhlo.gather(val1, val2, {}, [5], type).opcode).toBe('web.mhlo.gather');
  });

  it('should create regions ops', () => {
    const type = new TensorType([10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value(type, {} as Operation);
    const region = new Region();

    expect(mhlo.reduce([val1], [val2], [0], region, [type]).opcode).toBe('web.mhlo.reduce');
    expect(
      mhlo.reduceWindow([val1], [val2], [2], [1], [1], [1], [[0, 0]], region, [type]).opcode,
    ).toBe('web.mhlo.reduce_window');
    expect(mhlo.scatter(val1, val2, val1, region, {}, type).opcode).toBe('web.mhlo.scatter');
  });
});

describe('Lower ONNX to MHLO Pass', () => {
  it('should lower simple ONNX graph', () => {
    const graph = new Graph();
    graph.inputs = [
      new ValueInfo('A', [10, 10], 'float32'),
      new ValueInfo('B', [10, 10], 'float32'),
    ];
    graph.outputs = [new ValueInfo('C', [10, 10], 'float32')];
    graph.nodes = [new Node('Add', ['A', 'B'], ['C'])];

    const region = lowerONNXToMHLO(graph);
    expect(region.blocks.length).toBe(1);
    const block = region.blocks[0];

    expect(block.args.length).toBe(2); // A, B
    expect(block.operations.length).toBe(2); // Add, Return

    expect(block.operations[0].opcode).toBe('web.mhlo.add');
    expect(block.operations[1].opcode).toBe('web.mhlo.return');
  });
});
