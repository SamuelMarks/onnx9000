import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { KerasGenerator } from '../../../src/mmdnn/keras/generator';

describe('KerasGenerator Coverage', () => {
  it('should sanitize names correctly', () => {
    const graph = new Graph('test');
    const gen = new KerasGenerator(graph);
    expect(gen.sanitize('input:0')).toBe('input_0');
    expect(gen.sanitize('1abc')).toBe('v_1abc');
    expect(gen.sanitize('')).toBe('unnamed');
  });

  it('should get shapes from various sources', () => {
    const graph = new Graph('test');
    graph.addTensor(new Tensor('t1', [1, 2], 'float32'));
    graph.inputs.push(new ValueInfo('i1', [3, 4], 'float32'));
    graph.valueInfo.push(new ValueInfo('v1', [5, 6], 'float32'));

    const gen = new KerasGenerator(graph);
    expect(gen.getShape('t1')).toEqual([1, 2]);
    expect(gen.getShape('i1')).toEqual([3, 4]);
    expect(gen.getShape('v1')).toEqual([5, 6]);
    expect(gen.getShape('unknown')).toBeNull();
    expect(gen.getShape('')).toBeNull();
  });

  it('should generate NPY buffers for all supported dtypes', () => {
    const graph = new Graph('test');
    const gen = new KerasGenerator(graph);

    const dtypes = [
      'float32',
      'uint8',
      'int8',
      'uint16',
      'int16',
      'int32',
      'int64',
      'float16',
      'float64',
      'bool',
      'other',
    ];
    for (const dtype of dtypes) {
      const t = new Tensor('t', [2], dtype as any, true, false, new Uint8Array(8));
      const npy = gen.generateNpy(t);
      expect(npy.length).toBeGreaterThan(64);
      expect(npy[0]).toBe(0x93);
    }

    const tNull = new Tensor('t', [1], 'float32');
    expect(gen.generateNpy(tNull)).toBeDefined();
  });

  it('should generate source code for a complex graph', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('in', [1, 10], 'float32'));
    graph.addTensor(new Tensor('w', [10, 10], 'float32'));

    const n1 = new Node('Relu', ['in'], ['out1'], {}, 'relu1');
    const n2 = new Node('Add', ['out1', 'in'], ['out2'], {}, 'add1');
    const n3 = new Node('Unsupported', ['out2'], ['out3'], {}, 'unsupported1');
    graph.addNode(n1);
    graph.addNode(n2);
    graph.addNode(n3);
    graph.outputs.push(new ValueInfo('out3', [1, 10], 'float32'));

    const gen = new KerasGenerator(graph);
    const source = gen.generateSource();
    expect(source).toContain('import keras');
    expect(source).toContain('keras.layers.ReLU');
    expect(source).toContain('keras.layers.Add');
    expect(source).toContain('Fallback for Unsupported');
    expect(source).toContain('keras.Model');
  });

  it('should export weights as NPZ', () => {
    const graph = new Graph('test');
    graph.addTensor(new Tensor('w1', [1], 'float32', true, false, new Uint8Array(4)));
    const gen = new KerasGenerator(graph);
    const npz = gen.exportWeights();
    expect(npz).toBeInstanceOf(Uint8Array);
  });
});
