import { describe, it, expect } from 'vitest';
import { Region, Block, Operation, Value } from '../src/ir/core.js';
import { lowerLinalgToHAL } from '../src/passes/lower_linalg_to_hal.js';
import { TensorType } from '../src/dialects/web/tensor.js';
import * as memref from '../src/dialects/web/memref.js';

describe('Lower Linalg to HAL', () => {
  it('should lower memref.alloc, linalg.matmul, linalg.fill, and generic ops', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    const tensorType = new TensorType([2, 2], 'f32');
    const mType = new memref.MemRefType(tensorType.shape, tensorType.elementType);

    const allocOp = memref.alloc(mType);
    block.pushOperation(allocOp);

    const fillVal = new Value({ id: 'f32' });
    const fillOp = new Operation('web.linalg.fill', [fillVal, allocOp.results[0]!], [tensorType]);
    block.pushOperation(fillOp);

    const matmulOp = new Operation(
      'web.linalg.matmul',
      [allocOp.results[0]!, allocOp.results[0]!, allocOp.results[0]!],
      [tensorType],
    );
    block.pushOperation(matmulOp);

    const genericOp = new Operation('web.linalg.generic', [allocOp.results[0]!], [tensorType]);
    block.pushOperation(genericOp);

    const otherOp = new Operation('web.some.other.op', [allocOp.results[0]!], [{ id: 'i32' }]);
    block.pushOperation(otherOp);

    lowerLinalgToHAL(region);

    const opcodes = block.operations.map((o) => o.opcode);

    expect(opcodes).toContain('web.hal.device.get');
    expect(opcodes).toContain('web.hal.allocator.allocate');
    expect(opcodes).toContain('web.hal.command_buffer.create');
    expect(opcodes).toContain('web.hal.command_buffer.begin');
    expect(opcodes).toContain('web.hal.buffer.subspan');
    expect(opcodes).toContain('web.hal.executable.create');
    expect(opcodes).toContain('web.hal.command_buffer.dispatch');
    expect(opcodes).toContain('web.hal.command_buffer.fill_buffer');
    expect(opcodes).toContain('web.hal.command_buffer.end');
    expect(opcodes).toContain('web.hal.device.queue.submit');
    expect(opcodes).toContain('web.hal.device.queue.wait_idle');
  });
});
