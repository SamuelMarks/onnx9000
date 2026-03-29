/* eslint-disable */
// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { OnnxAstFormatter } from '../../src/core/OnnxAstFormatter';
import { VizGraph } from '../../src/core/OnnxAdapter';

describe('OnnxAstFormatter', () => {
  it('should format a VizGraph into a string', () => {
    const graph: VizGraph = {
      inputs: [{ name: 'input1', type: 'tensor(float)' }],
      outputs: [{ name: 'output1', type: 'tensor(float)' }],
      nodes: [
        {
          id: 'node1',
          name: 'Relu_1',
          opType: 'Relu',
          inputs: ['input1'],
          outputs: ['output1'],
          attributes: { alpha: 0.1 }
        }
      ]
    };

    const text = OnnxAstFormatter.format(graph);

    expect(text).toContain('// ONNX AST Structure:');
    expect(text).toContain('input: "input1"');
    expect(text).toContain('output: "output1"');
    expect(text).toContain('op_type: "Relu"');
    expect(text).toContain('name: "Relu_1"');
    expect(text).toContain('name: "alpha"');
    expect(text).toContain('value: 0.1');
    expect(text).toContain('type: "tensor(float)"');
  });

  it('should format correctly without inputs or outputs', () => {
    const graph: VizGraph = {
      inputs: [],
      outputs: [],
      nodes: [
        {
          id: 'node2',
          name: '',
          opType: 'Add',
          inputs: ['a', 'b'],
          outputs: ['c']
        }
      ]
    };

    const text = OnnxAstFormatter.format(graph);

    expect(text).toContain('op_type: "Add"');
    expect(text).not.toContain('name: ""');
    expect(text).not.toContain('attribute {');
  });

  it('should format when inputs and outputs are undefined', () => {
    const graph = {
      nodes: []
    } as object as VizGraph;

    const text = OnnxAstFormatter.format(graph);
    expect(text).toContain('graph {\n}\n');
  });
});
