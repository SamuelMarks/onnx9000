import { describe, it, expect } from 'vitest';
import { Region, Block, Operation, Value } from '../src/ir/core.js';
import { bufferizeLinalg } from '../src/passes/bufferize.js';
import { TensorType } from '../src/dialects/web/tensor.js';

describe('Bufferization Pass', () => {
  it('should bufferize web.tensor.empty, web.linalg.fill, and web.linalg.matmul', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    const tensorType = new TensorType([2, 2], 'f32');

    // empty tensor
    const emptyOp = new Operation('web.tensor.empty', [], [tensorType]);
    block.pushOperation(emptyOp);

    // dummy fill value
    const fillVal = new Value({ id: 'f32' });

    // fill
    const fillOp = new Operation('web.linalg.fill', [fillVal, emptyOp.results[0]!], [tensorType]);
    block.pushOperation(fillOp);

    // matmul
    const lhs = new Value(tensorType);
    const rhs = new Value(tensorType);
    const matmulOp = new Operation(
      'web.linalg.matmul',
      [lhs, rhs, fillOp.results[0]!],
      [tensorType],
    );
    block.pushOperation(matmulOp);

    // generic replacement for other tensor op
    const genericOp = new Operation('web.some.tensor.op', [matmulOp.results[0]!], [tensorType]);
    block.pushOperation(genericOp);

    // generic replacement for non-tensor op
    const nonTensorOp = new Operation(
      'web.some.other.op',
      [genericOp.results[0]!],
      [{ id: 'i32' }],
    );
    block.pushOperation(nonTensorOp);

    bufferizeLinalg(region);

    const opcodes = block.operations.map((o) => o.opcode);
    expect(opcodes).toContain('web.memref.alloc');
    expect(opcodes).toContain('web.linalg.fill');
    expect(opcodes).toContain('web.linalg.matmul');
    expect(opcodes).toContain('web.some.tensor.op');
    expect(opcodes).toContain('web.some.other.op');

    // Check operands of generic replacement for tensor
    const mappedGenericOp = block.operations.find((o) => o.opcode === 'web.some.tensor.op');
    expect(mappedGenericOp).toBeDefined();
    expect(mappedGenericOp!.operands.length).toBe(2); // one original + one alloc
  });
});
