import { describe, it, expect } from 'vitest';
import { _check_op_specific, ValidationContext, check_model } from '../src/checker.js';

describe('Checker Extra Coverage', () => {
  it('check_model should catch invalid ir_version', () => {
    const model = {
      ir_version: 11,
      opset_import: [{ domain: '', version: 17 }],
      graph: { inputs: [], initializers: [], nodes: [] },
    };
    const ctx = new ValidationContext();
    expect(() => check_model(model as any, ctx)).toThrow();
    expect(ctx.errors).toContain('Invalid ir_version');
  });

  it('_check_op_specific should handle Add with wrong inputs', () => {
    const ctx = new ValidationContext();
    _check_op_specific({ op_type: 'Add', inputs: ['x'], outputs: ['z'] }, ctx);
    expect(ctx.errors).toContain('Add requires 2 inputs');
  });

  it('_check_op_specific should handle If missing subgraphs', () => {
    const ctx = new ValidationContext();
    _check_op_specific({ op_type: 'If', inputs: ['cond'], outputs: ['z'] }, ctx);
    expect(ctx.errors).toContain('If requires subgraph attributes');
  });
});
