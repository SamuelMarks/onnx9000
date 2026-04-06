import { describe, it, expect, vi } from 'vitest';
import { OnnxScriptGenerator } from '../../src/mmdnn/onnxscript/generator.js';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('OnnxScriptGenerator Coverage Gaps', () => {
  const mockOnnxGraph = {
    name: 'TestGraph',
    nodes: [new Node('Relu', ['in'], ['out'])],
    inputs: [{ name: 'in', shape: [1, 3], dtype: 'float32' }],
    outputs: [{ name: 'out', shape: [1, 3], dtype: 'float32' }],
    tensors: { w: new Tensor('w', [1], 'float32', true, false, new Uint8Array(4)) },
    initializers: ['w'],
    valueInfo: [],
  };

  it('should cover generator branches and complex nodes', () => {
    const gen = new OnnxScriptGenerator(mockOnnxGraph as Object);
    const code = gen.generate();
    expect(code).toContain('import onnxscript');

    // Test with complex node types
    const complexGraph = {
      ...mockOnnxGraph,
      nodes: [new Node('Conv', ['in', 'w'], ['c']), new Node('Add', ['c', 'in'], ['out'])],
    };
    const gen2 = new OnnxScriptGenerator(complexGraph as Object);
    expect(gen2.generate()).toContain('op.Conv');
  });

  it('should cover naming and shape fallbacks', () => {
    const edgeGraph = {
      name: '',
      nodes: [new Node('Identity', ['1'], ['2'])],
      inputs: [],
      outputs: [],
      tensors: {},
      initializers: [],
      valueInfo: [],
    };
    const gen = new OnnxScriptGenerator(edgeGraph as Object);
    expect(gen.generate()).toContain('def unnamed');
  });
});
