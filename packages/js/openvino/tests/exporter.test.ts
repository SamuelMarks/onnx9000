import { describe, it, expect } from 'vitest';
import { exportModel } from '../src/api.js';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('OpenVinoExporter', () => {
  it('should export a simple graph to XML/BIN', () => {
    const graph = new Graph('TestModel');
    graph.addNode(new Node('Add', ['X', 'Y'], ['Z'], {}, 'add_node'));
    graph.inputs.push({ name: 'X', shape: [1, 3], dtype: 'float32' });
    graph.inputs.push({ name: 'Y', shape: [1, 3], dtype: 'float32' });
    graph.outputs.push({ name: 'Z', shape: [1, 3], dtype: 'float32' });

    const result = exportModel(graph);
    expect(result.xml).toContain('<net name=\"TestModel\"');
    expect(result.xml).toContain('<layer id=\"0\" name=\"X\" type=\"Parameter\"');
    expect(result.bin).toBeDefined();
  });

  it('should handle constants and basic math', () => {
    const graph = new Graph('ConstModel');
    const t = new Tensor('W', [1], 'float32', true);
    t.data = new Uint8Array(new Float32Array([42]).buffer);
    graph.tensors['W'] = t;
    graph.initializers.push('W');

    graph.addNode(new Node('Mul', ['X', 'W'], ['Y'], {}, 'mul_node'));
    graph.inputs.push({ name: 'X', shape: [1], dtype: 'float32' });
    graph.outputs.push({ name: 'Y', shape: [1], dtype: 'float32' });

    const result = exportModel(graph);
    expect(result.xml).toContain('type=\"Const\"');
    expect(result.xml).toContain('type=\"Multiply\"');
  });

  it('should support FP16 compression option', () => {
    const graph = new Graph('Fp16Model');
    graph.inputs.push({ name: 'X', shape: [1], dtype: 'float32' });
    graph.addNode(new Node('Relu', ['X'], ['Y'], {}, 'relu'));
    graph.outputs.push({ name: 'Y', shape: [1], dtype: 'float32' });

    const result = exportModel(graph, { compressToFp16: true });
    expect(result.xml).toBeDefined();
  });
});
