import { describe, expect, it } from 'vitest';
import { type DType, SparseTensor, Tensor } from '../src/ir/tensor.js';
import { getTypedArray, sparseToDense, unpackData } from '../src/sparse.js';

describe('Sparse Extra Coverage Final', () => {
  it('unpackData should handle int64 and fallback', () => {
    const data = new BigInt64Array([1n, 2n]);
    const t = new Tensor('t', [2], 'int64', false, true, new Uint8Array(data.buffer));
    const unpacked = unpackData(t);
    expect(unpacked).toEqual([1n, 2n]);

    // Fallback for non-Uint8Array data
    const t2 = new Tensor('t2', [2], 'float32', false, true, new Float32Array([1, 2]));
    const unpacked2 = unpackData(t2);
    expect(unpacked2).toEqual([1, 2]);
  });

  it('sparseToDense should handle missing tensors', () => {
    const st = new SparseTensor('st', [2, 2], 'COO', null, null, null, null, 'float32');
    const dense = sparseToDense(st);
    expect(dense.data).toBeNull();
  });

  it('getTypedArray fallback', () => {
    const arr = getTypedArray('unknown' as DType, 5);
    expect(arr).toBeInstanceOf(Float32Array);
  });
});
