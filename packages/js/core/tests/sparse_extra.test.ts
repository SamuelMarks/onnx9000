import { describe, expect, it } from 'vitest';
import { SparseTensor, Tensor } from '../src/ir/tensor.js';
import { sparseToDense, unpackData } from '../src/sparse.js';

describe('Sparse Extra Coverage', () => {
  it('unpackData should handle int64', () => {
    const data = new BigInt64Array([1n, 2n]);
    const t = new Tensor('t', [2], 'int64', false, true, new Uint8Array(data.buffer));
    const unpacked = unpackData(t);
    expect(unpacked).toEqual([1n, 2n]);
  });

  it('sparseToDense should handle missing tensors', () => {
    const st = new SparseTensor('st', [2], 'COO', null, null);
    const dense = sparseToDense(st);
    expect(dense.data).toBeNull();
  });
});
