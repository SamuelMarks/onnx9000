import { describe, expect, it } from 'vitest';
import { ONNXToPyTorchVisitor } from '../src/codegen/pytorch.js';
import { Graph } from '../src/ir/graph.js';

describe('ONNXToPyTorchVisitor', () => {
  it('should generate code', () => {
    const v = new ONNXToPyTorchVisitor(new Graph('test'));
    const code = v.generate();
    expect(code).toContain('import torch');
  });
});
