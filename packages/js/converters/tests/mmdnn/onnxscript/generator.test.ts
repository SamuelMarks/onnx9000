import { describe, it, expect } from 'vitest';
import { OnnxScriptGenerator } from '../../../src/mmdnn/onnxscript/generator.js';

describe('OnnxScriptGenerator', () => {
  it('should generate', () => {
    const gen = new OnnxScriptGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      nodes: [],
      attributes: {},
    } as any);
    const code = gen.generate();
    expect(code).toContain('import onnxscript');
  });
});
