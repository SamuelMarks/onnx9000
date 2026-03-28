import { describe, it, expect } from 'vitest';
import { Tensor, SparseTensor } from '../src/ir/tensor.js';

describe('IR Tensor Gaps 2', () => {
  it('SparseTensor copy with all tensors', () => {
    const v = new Tensor('v', [1], 'float32');
    const i = new Tensor('i', [1], 'int32');
    const r = new Tensor('r', [1], 'int32');
    const c = new Tensor('c', [1], 'int32');
    const st = new SparseTensor('st', [1, 1], 'COO', v, i, r, c, [1, 1]);
    const sc = st.copy() as SparseTensor;
    expect(sc.valuesTensor?.name).toBe('v');
    expect(sc.indicesTensor?.name).toBe('i');
    expect(sc.rowPtrTensor?.name).toBe('r');
    expect(sc.colIndicesTensor?.name).toBe('c');
    expect(sc.blockDims).toEqual([1, 1]);
  });
});
