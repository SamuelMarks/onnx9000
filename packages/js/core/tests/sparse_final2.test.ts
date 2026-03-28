import { describe, it, expect, vi } from 'vitest';
import { Tensor, SparseTensor } from '../src/ir/tensor.js';
import {
  getTypedArray,
  unpackData,
  denseToCoo,
  denseToCsr,
  denseToCsc,
  denseToBsr,
  sparseToCoo,
  sparseToDense,
} from '../src/sparse.js';

describe('Sparse Operations Final', () => {
  it('unpackData more branches', () => {
    const data32 = new Int32Array([1, 2]);
    const t3 = new Tensor('t3', [2], 'int32', true, false, new Uint8Array(data32.buffer));
    expect(unpackData(t3)).toEqual([1, 2]);

    const tFloat = new Tensor('tF', [2], 'float32', true, false, new Float32Array([1.1, 2.2]));
    tFloat.data = new Uint8Array(new Float32Array([1.1, 2.2]).buffer);
    const res = unpackData(tFloat) as number[];
    expect(res[0]).toBeCloseTo(1.1);
  });

  it('sparseToCoo more branches', () => {
    // Create a mock CSC tensor
    const vals = new Tensor('vals', [1], 'float32', false, false, new Float32Array([42.0]));
    const rowIdx = new Tensor('rowIdx', [1], 'int64', false, false, new Int32Array([0]));
    const colPtr = new Tensor('colPtr', [3], 'int64', false, false, new Int32Array([0, 0, 1]));

    const st = new SparseTensor('st', [2, 2], 'CSC', vals, null, colPtr, rowIdx);
    const coo = sparseToCoo(st);
    expect(coo.format).toBe('COO');

    // Create a mock BSR tensor
    const bsrVals = new Tensor('bVals', [1], 'float32', false, false, new Float32Array([42.0]));
    const bsrColIdx = new Tensor('bColIdx', [1], 'int64', false, false, new Int32Array([1]));
    const bsrRowPtr = new Tensor('bRowPtr', [3], 'int64', false, false, new Int32Array([0, 0, 1]));

    const bsr = new SparseTensor('bsr', [2, 2], 'BSR', bsrVals, null, bsrRowPtr, bsrColIdx, [1, 1]);
    const bsrCoo = sparseToCoo(bsr);
    expect(bsrCoo.format).toBe('COO');
  });
});
