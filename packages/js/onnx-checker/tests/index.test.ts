import { describe, expect, it } from 'vitest';
import { Onnxchecker } from '../src/index.js';

describe('Onnxchecker', () => {
  it('should validate valid and invalid ONNX models', () => {
    const checker = new Onnxchecker();
    expect(checker.run()).toBe('[onnx-checker] processed');

    const errors = checker.validate({
      irVersion: 2,
      opsetVersions: [],
      inputs: ['a'],
      nodes: [{ opType: 'Add', inputs: ['a', 'missing_b'], outputs: ['c'] }],
    });
    expect(errors).toEqual([
      'Unsupported irVersion: 2',
      'Missing opset import',
      "Unresolved input 'missing_b' in node 'Add'",
    ]);
  });
});
