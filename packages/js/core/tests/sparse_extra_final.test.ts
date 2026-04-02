import { describe, it, expect } from 'vitest';
import { unpackData, sparseToDense, getTypedArray } from '../src/sparse.js';
import { Tensor } from '../src/ir/tensor.js';

describe('Sparse Extra Coverage Final', () => {
  it('unpackData should handle int64 and fallback', () => {
    const data = new BigInt64Array([1n, 2n]);
    const t = new Tensor('t', [2], 'int64', false, true, new Uint8Array(data.buffer));
    const unpacked = unpackData(t);
    expect(unpacked).toEqual([1n, 2n]);

    // Fallback for non-Uint8Array data
    const t2 = new Tensor('t2', [2], 'float32', false, true, new Float32Array([1, 2]) as any);
    const unpacked2 = unpackData(t2);
    expect(unpacked2).toEqual([1, 2]);
  });

  it('sparseToDense should handle missing tensors', () => {
    // Use a plain object mock if SparseTensor constructor is problematic
    const st = {
      name: 'st',
      shape: [2, 2],
      format: 'COO',
      valuesTensor: null,
      indicesTensor: null,
      dtype: 'float32',
    };
    const dense = sparseToDense(st as any);
    expect(dense.data).toBeNull();
  });

  it('getTypedArray fallback', () => {
    const arr = getTypedArray('unknown' as any, 5);
    expect(arr).toBeInstanceOf(Float32Array);
  });
});
