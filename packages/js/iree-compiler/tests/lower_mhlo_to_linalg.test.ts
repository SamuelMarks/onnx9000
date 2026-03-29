import { describe, it, expect } from 'vitest';
import { Region, Block, Operation, Value } from '../src/ir/core.js';
import { lowerMHLOToLinalg } from '../src/passes/lower_mhlo_to_linalg.js';
import { TensorType } from '../src/dialects/web/tensor.js';

describe('Lower MHLO To Linalg Pass', () => {
  it('should lower web.mhlo.add/subtract and web.mhlo.dot to linalg equivalents', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    const tensorType = new TensorType([2, 2], 'f32');

    const a = new Value(tensorType);
    const b = new Value(tensorType);

    // add
    const addOp = new Operation('web.mhlo.add', [a, b], [tensorType]);
    block.pushOperation(addOp);

    // sub
    const subOp = new Operation('web.mhlo.subtract', [a, b], [tensorType]);
    block.pushOperation(subOp);

    // dot
    const dotOp = new Operation('web.mhlo.dot', [a, b], [tensorType]);
    block.pushOperation(dotOp);

    // other
    const otherOp = new Operation('web.mhlo.other', [a], [tensorType]);
    block.pushOperation(otherOp);

    lowerMHLOToLinalg(region);

    const opcodes = block.operations.map((o) => o.opcode);

    // add/sub both generate empty and generic
    expect(opcodes.filter((x) => x === 'web.tensor.empty').length).toBe(3);
    expect(opcodes.filter((x) => x === 'web.linalg.generic').length).toBe(2);

    // dot generates empty, constant, fill, matmul
    expect(opcodes).toContain('web.mhlo.constant');
    expect(opcodes).toContain('web.linalg.fill');
    expect(opcodes).toContain('web.linalg.matmul');

    // other passes through
    expect(opcodes).toContain('web.mhlo.other');
  });
});
