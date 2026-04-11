import { describe, vi, expect, it } from 'vitest';
import { Graph } from '@onnx9000/core';
import { PaddleMapper, translatePaddleShape } from '../../src/mmdnn/paddle/mapper.js';
import { PaddleParser } from '../../src/mmdnn/paddle/parser.js';

describe('PaddleParser', () => {
  it('should parse model JSON', () => {
    const parser = new PaddleParser();
    const result = parser.parseModel('{"name": "test"}');
    expect(result.name).toBe('test');

    const objResult = parser.parseModel({ name: 'test_obj' });
    expect(objResult.name).toBe('test_obj');
  });

  it('should parse weights stub', () => {
    const parser = new PaddleParser();
    const buffer = new Uint8Array([1, 2, 3]);
    const result = parser.parseWeights(buffer);
    expect(result.byteLength).toBe(3);
  });
});

describe('PaddleMapper', () => {
  it('should map conv2d', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();

    // with inputs and attrs
    const layer = {
      type: 'conv2d',
      name: 'conv',
      inputs: { Input: ['x'], Filter: ['w'], Bias: ['b'] },
      outputs: { Output: ['out'] },
      attrs: { paddings: [1, 1, 1, 1], strides: [2, 2], dilations: [1, 1], groups: 1 },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('Conv');
    expect(node?.inputs).toEqual(['x', 'w', 'b']);
    expect(node?.outputs).toEqual(['out']);
    expect(node?.attributes.pads?.value).toEqual([1, 1, 1, 1]);
    expect(node?.attributes.strides?.value).toEqual([2, 2]);

    // missing attrs and inputs
    const layer2 = { type: 'conv2d' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2).toHaveLength(1);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['conv2d_out']);
    expect(nodes2[0]?.attributes.pads?.value).toEqual([0, 0, 0, 0]);
  });

  it('should map pool2d (max)', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'pool2d',
      name: 'pool_max',
      inputs: { X: ['x'] },
      outputs: { Out: ['out'] },
      attrs: {
        pooling_type: 'max',
        paddings: [0, 0, 0, 0],
        strides: [2, 2],
        ksize: [2, 2],
        ceil_mode: false,
      },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('MaxPool');
    expect(node?.inputs).toEqual(['x']);
    expect(node?.outputs).toEqual(['out']);
    expect(node?.attributes.kernel_shape?.value).toEqual([2, 2]);
    expect(node?.attributes.ceil_mode?.value).toBe(0);

    const layer2 = { type: 'pool2d' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.opType).toBe('MaxPool');
  });

  it('should map pool2d (avg)', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'pool2d',
      name: 'pool_avg',
      inputs: { X: ['x'] },
      outputs: { Out: ['out'] },
      attrs: {
        pooling_type: 'avg',
        paddings: [0, 0, 0, 0],
        strides: [1, 1],
        ksize: [3, 3],
        ceil_mode: true,
      },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('AveragePool');
    expect(node?.inputs).toEqual(['x']);
    expect(node?.outputs).toEqual(['out']);
    expect(node?.attributes.kernel_shape?.value).toEqual([3, 3]);
    expect(node?.attributes.ceil_mode?.value).toBe(1);
  });

  it('should map elementwise_add', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'elementwise_add',
      name: 'add',
      inputs: { X: ['x'], Y: ['y'] },
      outputs: { Out: ['out'] },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('Add');
    expect(node?.inputs).toEqual(['x', 'y']);
    expect(node?.outputs).toEqual(['out']);

    const layer2 = { type: 'elementwise_add' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['add_out']);
  });

  it('should map relu', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'relu',
      name: 'relu',
      inputs: { X: ['x'] },
      outputs: { Out: ['out'] },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('Relu');
    expect(node?.inputs).toEqual(['x']);
    expect(node?.outputs).toEqual(['out']);

    const layer2 = { type: 'relu' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['relu_out']);
  });

  it('should map batch_norm', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'batch_norm',
      name: 'bn',
      inputs: { X: ['x'], Scale: ['scale'], Bias: ['bias'], Mean: ['mean'], Variance: ['var'] },
      outputs: { Y: ['out'] },
      attrs: { epsilon: 1e-5, momentum: 0.9 },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('BatchNormalization');
    expect(node?.inputs).toEqual(['x', 'scale', 'bias', 'mean', 'var']);
    expect(node?.outputs).toEqual(['out']);
    expect(node?.attributes.epsilon?.value).toBeCloseTo(1e-5);

    const layer2 = { type: 'batch_norm' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['batch_norm_out']);
  });

  it('should map mul', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'mul',
      name: 'mul',
      inputs: { X: ['x'], Y: ['y'] },
      outputs: { Out: ['out'] },
      attrs: { x_num_col_dims: 1, y_num_col_dims: 1 },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('MatMul');
    expect(node?.inputs).toEqual(['x', 'y']);
    expect(node?.outputs).toEqual(['out']);

    const layer2 = { type: 'mul' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['mul_out']);
  });

  it('should map concat', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'concat',
      name: 'concat',
      inputs: { X: ['x1', 'x2'] },
      outputs: { Out: ['out'] },
      attrs: { axis: 1 },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('Concat');
    expect(node?.inputs).toEqual(['x1', 'x2']);
    expect(node?.outputs).toEqual(['out']);
    expect(node?.attributes.axis?.value).toBe(1);

    const layer2 = { type: 'concat' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual(['concat_out']);
  });

  it('should map split', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'split',
      name: 'split',
      inputs: { X: ['x'] },
      outputs: { Out: ['out1', 'out2'] },
      attrs: { axis: 1, num_or_sections: [2, 3] },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(1);
    const node = nodes[0];
    expect(node?.opType).toBe('Split');
    expect(node?.inputs).toEqual(['x']);
    expect(node?.outputs).toEqual(['out1', 'out2']);
    expect(node?.attributes.axis?.value).toBe(1);
    expect(node?.attributes.split?.value).toEqual([2, 3]);

    const layer2 = { type: 'split' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2[0]?.inputs).toEqual([]);
    expect(nodes2[0]?.outputs).toEqual([]);
  });

  it('should map matmul with transpose', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = {
      type: 'matmul',
      name: 'matmul',
      inputs: { X: ['x'], Y: ['y'] },
      outputs: { Out: ['out'] },
      attrs: { transpose_X: true, transpose_y: true },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(3);
    expect(nodes[0]?.opType).toBe('Transpose');
    expect(nodes[0]?.inputs).toEqual(['x']);
    expect(nodes[0]?.outputs).toEqual(['x_trans']);

    expect(nodes[1]?.opType).toBe('Transpose');
    expect(nodes[1]?.inputs).toEqual(['y']);
    expect(nodes[1]?.outputs).toEqual(['y_trans']);

    expect(nodes[2]?.opType).toBe('MatMul');
    expect(nodes[2]?.inputs).toEqual(['x_trans', 'y_trans']);
    expect(nodes[2]?.outputs).toEqual(['out']);

    const layer2 = { type: 'matmul' };
    const nodes2 = mapper.map(layer2, graph);
    expect(nodes2).toHaveLength(1);
    expect(nodes2[0]?.opType).toBe('MatMul');
    expect(nodes2[0]?.inputs).toEqual([undefined, undefined]);
    expect(nodes2[0]?.outputs).toEqual(['matmul_out']);
  });

  it('should handle unknown layer type', () => {
    const mapper = new PaddleMapper();
    const graph = new Graph();
    const layer = { type: 'unknown_op' };

    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const nodes = mapper.map(layer, graph);
    expect(nodes).toHaveLength(0);
    expect(consoleSpy).toHaveBeenCalledWith('Unsupported Paddle layer type: unknown_op');
    consoleSpy.mockRestore();
  });

  it('should translate Paddle shape to ONNX dynamic shape', () => {
    const shape = translatePaddleShape([-1, 3, 224, 224]);
    expect(shape).toEqual(['dynamic', 3, 224, 224]);
  });
});
