import { describe, it, expect } from 'vitest';
import { TFMapper } from '../../../src/mmdnn/tensorflow/mapper.js';
import { Graph } from '@onnx9000/core';

describe('TFMapper', () => {
  it('should map a Placeholder node', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map({ name: 'input1', op: 'Placeholder', input: [], attr: {} }, graph);
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Identity');
    expect(nodes[0].inputs).toEqual(['input1_input_dummy']);
  });

  it('should map a Const node to an initializer', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'weights1',
        op: 'Const',
        input: [],
        attr: {
          value: { tensor: { dtype: 'DT_FLOAT', shape: [3, 3, 1, 32] } },
        },
      },
      graph,
    );
    expect(nodes.length).toBe(0);
    expect(graph.initializers).toContain('weights1');
    expect(graph.tensors['weights1']).toBeDefined();
    expect(graph.tensors['weights1'].shape).toEqual([3, 3, 1, 32]);
  });

  it('should handle Const without tensor shape', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'weights2',
        op: 'Const',
        input: [],
        attr: {},
      },
      graph,
    );
    expect(nodes.length).toBe(0);
    expect(graph.tensors['weights2'].shape).toEqual([1]);
  });

  it('should map Conv2D to Conv with adjusted strides and padding', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'conv1',
        op: 'Conv2D',
        input: ['in', 'w'],
        attr: {
          strides: { list: { i: [1, 2, 2, 1] } },
          padding: { s: 'SAME' },
        },
      },
      graph,
    );
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Conv');
    expect(nodes[0].attributes['strides'].value).toEqual([2, 2]);
    expect(nodes[0].attributes['auto_pad'].value).toBe('SAME_UPPER');
  });

  it('should map VALID padding for Conv2D', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'conv1',
        op: 'Conv2D',
        input: ['in', 'w'],
        attr: { padding: { s: 'VALID' } },
      },
      graph,
    );
    expect(nodes[0].attributes['auto_pad'].value).toBe('VALID');
  });

  it('should map MaxPool to MaxPool with adjusted strides and kernel shape', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'pool1',
        op: 'MaxPool',
        input: ['in'],
        attr: {
          ksize: { list: { i: [1, 3, 3, 1] } },
          strides: { list: { i: [1, 2, 2, 1] } },
          padding: { s: 'SAME' },
        },
      },
      graph,
    );
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('MaxPool');
    expect(nodes[0].attributes['strides'].value).toEqual([2, 2]);
    expect(nodes[0].attributes['kernel_shape'].value).toEqual([3, 3]);
    expect(nodes[0].attributes['auto_pad'].value).toBe('SAME_UPPER');
  });

  it('should map VALID padding for MaxPool', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'pool1',
        op: 'MaxPool',
        input: ['in'],
        attr: { padding: { s: 'VALID' } },
      },
      graph,
    );
    expect(nodes[0].attributes['auto_pad'].value).toBe('VALID');
  });

  it('should map Relu6 to Relu', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map({ name: 'relu1', op: 'Relu6', input: ['in'], attr: {} }, graph);
    expect(nodes[0].opType).toBe('Relu');
  });

  it('should handle integer and float attributes', () => {
    const mapper = new TFMapper();
    const graph = new Graph('test');
    const nodes = mapper.map(
      {
        name: 'node1',
        op: 'SomeOp',
        input: ['in'],
        attr: {
          my_int: { i: 42 },
          my_float: { f: 3.14 },
          my_shape: { shape: [1, 2, 3] },
        },
      },
      graph,
    );
    expect(nodes[0].attributes['my_int'].value).toBe(42);
    expect(nodes[0].attributes['my_float'].value).toBe(3.14);
    expect(nodes[0].attributes['my_shape'].value).toEqual([1, 2, 3]);
  });
});
