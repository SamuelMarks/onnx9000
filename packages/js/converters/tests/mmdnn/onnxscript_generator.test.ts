import { describe, it, expect } from 'vitest';
import { OnnxScriptGenerator } from '../../src/mmdnn/onnxscript/generator.js';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('OnnxScriptGenerator', () => {
  it('should generate basic ONNXScript code', () => {
    const graph = new Graph('TestGraph');
    graph.inputs = [{ name: 'input_1', type: 'tensor', shape: [1, 3] }];
    graph.outputs = [{ name: 'output_1', type: 'tensor', shape: [1, 3] }];

    const node = new Node('Relu', ['input_1'], ['output_1']);
    graph.nodes.push(node);

    const generator = new OnnxScriptGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('import onnxscript');
    expect(code).toContain('def model(input_1: FLOAT[...]):');
    expect(code).toContain('output_1 = op.Relu(input_1)');
    expect(code).toContain('return output_1');
  });

  it('should handle attributes and multiple outputs', () => {
    const graph = new Graph('AttrGraph');
    graph.inputs = [{ name: 'x', type: 'tensor', shape: [1] }];
    graph.outputs = [{ name: 'y', type: 'tensor', shape: [1] }];

    const node = new Node('Add', ['x', 'x'], ['y'], {
      alpha: { name: 'alpha', value: 1.0, attrType: 'FLOAT' },
    });
    graph.nodes.push(node);

    const generator = new OnnxScriptGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('y = op.Add(x, x, alpha=1)');
  });

  it('should handle empty graph', () => {
    const graph = new Graph('Empty');
    const generator = new OnnxScriptGenerator(graph);
    const code = generator.generate();
    expect(code).toContain('def model(input: FLOAT[...]):');
    expect(code).toContain('pass');
  });
});
