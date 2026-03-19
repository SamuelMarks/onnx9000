import { describe, it, expect } from 'vitest';
import { Type, Value, Region, Operation } from '../src/ir/core.js';
import * as linalg from '../src/dialects/web/linalg.js';
import * as memref from '../src/dialects/web/memref.js';
import { TensorType } from '../src/dialects/web/tensor.js';

describe('Web Linalg Dialect', () => {
  it('should create linalg ops', () => {
    const type = new TensorType([10, 10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value(type, {} as Operation);
    const val3 = new Value(type, {} as Operation);

    expect(linalg.matmul(val1, val2, val3, type).opcode).toBe('web.linalg.matmul');
    expect(linalg.batchMatmul(val1, val2, val3, type).opcode).toBe('web.linalg.batch_matmul');
    expect(linalg.conv2dNhwcHwcf(val1, val2, val3, [1, 1], [1, 1], type).opcode).toBe(
      'web.linalg.conv_2d_nhwc_hwcf',
    );
    expect(linalg.poolingNhwcMax(val1, val2, val3, [1, 1], [1, 1], type).opcode).toBe(
      'web.linalg.pooling_nhwc_max',
    );
    expect(linalg.fill(val1, val2, type).opcode).toBe('web.linalg.fill');
    expect(linalg.yieldOp([val1]).opcode).toBe('web.linalg.yield');
  });

  it('should create generic op', () => {
    const type = new TensorType([10, 10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const region = new Region();
    const map = linalg.AffineMap.getMinorIdentity(2);

    const gen = linalg.generic([val1], [val1], [map, map], ['parallel', 'parallel'], region, [
      type,
    ]);
    expect(gen.opcode).toBe('web.linalg.generic');
    expect(gen.attributes.iterator_types).toEqual(['parallel', 'parallel']);
  });
});

describe('Web Memref Dialect', () => {
  it('should create memref ops', () => {
    const type = new memref.MemRefType([10, 10], 'float32');
    const val1 = new Value(type, {} as Operation);
    const val2 = new Value({ id: 'i32' }, {} as Operation);

    expect(memref.alloc(type).opcode).toBe('web.memref.alloc');
    expect(memref.dealloc(val1).opcode).toBe('web.memref.dealloc');
    expect(memref.load(val1, [val2, val2], { id: 'float32' }).opcode).toBe('web.memref.load');
    expect(memref.store(val1, val1, [val2, val2]).opcode).toBe('web.memref.store');
  });
});
