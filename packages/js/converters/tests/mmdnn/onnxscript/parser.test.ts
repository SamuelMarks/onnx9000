import { describe, it, expect } from 'vitest';
import { OnnxScriptParser } from '../src/mmdnn/onnxscript/parser.js';

describe('OnnxScriptParser', () => {
  it('should parse', () => {
    const parser = new OnnxScriptParser();
    const g = parser.parseScript('def test(x: FLOAT):\n  y = op.Relu(x)\n  return y');
    expect(g.inputs.length).toBe(1);
    expect(g.nodes[0].opType).toBe('Relu');
  });
});
