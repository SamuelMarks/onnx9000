import { describe, it, expect } from 'vitest';
import { OnnxScriptParser } from '../../src/mmdnn/onnxscript/parser.js';

describe('MMDNN - OnnxScript Importer', () => {
  it('should parse basic ONNXScript Python code', () => {
    const parser = new OnnxScriptParser();
    const mockCode = `
import onnxscript
from onnxscript import opset15 as op
from onnxscript import FLOAT

@onnxscript.script()
def mlp(X: FLOAT[10, 10], W1: FLOAT[10, 20], B1: FLOAT[20]) -> FLOAT[10, 20]:
    H1 = op.MatMul(X, W1)
    H1_bias = op.Add(H1, B1)
    H1_relu = op.Relu(H1_bias)
    return H1_relu
`;

    const graph = parser.parseScript(mockCode);

    // Inputs
    expect(graph.inputs.length).toBe(3);
    expect(graph.inputs[0].name).toBe('X');
    expect(graph.inputs[0].shape).toEqual([10, 10]);

    // Nodes
    expect(graph.nodes.length).toBe(3);
    expect(graph.nodes[0].opType).toBe('MatMul');
    expect(graph.nodes[0].inputs).toEqual(['X', 'W1']);
    expect(graph.nodes[0].outputs).toEqual(['H1']);

    // Outputs
    expect(graph.outputs.length).toBe(1);
    expect(graph.outputs[0].name).toBe('H1_relu');
  });

  it('should handle un-annotated or empty functions gracefully', () => {
    const parser = new OnnxScriptParser();
    const graph = parser.parseScript('def empty():\n  pass\n');
    expect(graph.inputs.length).toBe(0);
    expect(graph.nodes.length).toBe(0);
  });
});
