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

describe('Sparse Operations', () => {
  it('getTypedArray coverage', () => {
    expect(getTypedArray('float32', 10)).toBeInstanceOf(Float32Array);
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(getTypedArray('float64', 10)).toBeInstanceOf(Float32Array);
    expect(spy).toHaveBeenCalled();
    expect(getTypedArray('int8', 10)).toBeInstanceOf(Int8Array);
    expect(getTypedArray('int16', 10)).toBeInstanceOf(Int16Array);
    expect(getTypedArray('int32', 10)).toBeInstanceOf(Int32Array);
    expect(getTypedArray('uint8', 10)).toBeInstanceOf(Uint8Array);
    expect(getTypedArray('uint16', 10)).toBeInstanceOf(Uint16Array);
    expect(getTypedArray('uint32', 10)).toBeInstanceOf(Uint32Array);
    expect(getTypedArray('int64', 10)).toBeInstanceOf(BigInt64Array);
    expect(getTypedArray('uint64', 10)).toBeInstanceOf(BigUint64Array);
    expect(getTypedArray('float16', 10)).toBeInstanceOf(Uint16Array);
    expect(getTypedArray('bool', 10)).toBeInstanceOf(Uint8Array);
    // @ts-ignore
    expect(getTypedArray('unknown', 10)).toBeInstanceOf(Float32Array);
  });

  it('unpackData coverage', () => {
    const data = new Float32Array([1, 0, 3, 0]);
    const t = new Tensor('t', [4], 'float32', true, false, new Uint8Array(data.buffer));
    expect(unpackData(t)).toEqual([1, 0, 3, 0]);

    const t2 = new Tensor('t2', [0], 'float32');
    expect(unpackData(t2)).toEqual([]);

    const data32 = new Int32Array([1, 2]);
    const t3 = new Tensor('t3', [2], 'int32', true, false, new Uint8Array(data32.buffer));
    expect(unpackData(t3)).toEqual([1, 2]);

    const data64 = new BigInt64Array([1n, 2n]);
    const t4 = new Tensor('t4', [2], 'int64', true, false, new Uint8Array(data64.buffer));
    expect(unpackData(t4)).toEqual([1n, 2n]);

    const t5 = new Tensor('t5', [2], 'int8', true, false, new Int8Array([5, 6]));
    expect(unpackData(t5)).toEqual([5, 6]);
  });

  it('denseToCoo coverage', () => {
    const t = new Tensor('t', [2, 2], 'float32', true, false, new Float32Array([1, 0, 0, 4]));
    const st = denseToCoo(t);
    expect(st.format).toBe('COO');
    expect(unpackData(st.valuesTensor!)).toEqual([1, 4]);
    expect(unpackData(st.indicesTensor!)).toEqual([0, 3]);
  });

  it('denseToCsr coverage', () => {
    const t = new Tensor('t', [2, 2], 'float32', true, false, new Float32Array([1, 0, 0, 4]));
    const st = denseToCsr(t);
    expect(st.format).toBe('CSR');
    expect(unpackData(st.valuesTensor!)).toEqual([1, 4]);
    expect(unpackData(st.colIndicesTensor!)).toEqual([0, 1]);
    expect(unpackData(st.rowPtrTensor!)).toEqual([0, 1, 2]);

    const t1d = new Tensor('t1d', [4], 'float32', true, false, new Float32Array([1, 0, 3, 0]));
    expect(denseToCsr(t1d).format).toBe('COO');

    const tLarge = new Tensor(
      'tLarge',
      [2, 2],
      'float32',
      true,
      false,
      new Float32Array([1, 0, 0, 4]),
    );
    // small model warning
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    denseToCsr(tLarge);
    expect(spy).toHaveBeenCalled();

    const tTooLarge = new Tensor('tTooLarge', [2147483648, 1], 'float32');
    expect(() => denseToCsr(tTooLarge)).toThrow();
  });

  it('denseToCsc coverage', () => {
    const t = new Tensor('t', [2, 2], 'float32', true, false, new Float32Array([1, 0, 0, 4]));
    const st = denseToCsc(t);
    expect(st.format).toBe('CSC');
    expect(unpackData(st.valuesTensor!)).toEqual([1, 4]);
    expect(unpackData(st.colIndicesTensor!)).toEqual([0, 1]);
    expect(unpackData(st.rowPtrTensor!)).toEqual([0, 1, 2]);

    const t1d = new Tensor('t1d', [4], 'float32', true, false, new Float32Array([1, 0, 3, 0]));
    expect(denseToCsc(t1d).format).toBe('COO');
  });

  it('denseToBsr coverage', () => {
    const t = new Tensor('t', [4, 4], 'float32', true, false, new Float32Array(16).fill(0));
    const data = unpackData(t) as number[];
    data[0] = 1;
    data[1] = 2;
    data[4] = 3;
    data[5] = 4;
    t.data = new Float32Array(data);

    const st = denseToBsr(t, [2, 2]);
    expect(st.format).toBe('BSR');
    expect(unpackData(st.valuesTensor!).length).toBe(4);
    expect(unpackData(st.rowPtrTensor!)).toEqual([0, 1, 1]);

    const t1d = new Tensor('t1d', [4], 'float32');
    expect(denseToBsr(t1d, [2, 2]).format).toBe('COO');

    const tOdd = new Tensor('tOdd', [3, 3], 'float32');
    expect(denseToBsr(tOdd, [2, 2]).format).toBe('COO');
  });

  it('sparseToCoo and sparseToDense coverage', () => {
    const t = new Tensor('t', [2, 2], 'float32', true, false, new Float32Array([1, 0, 0, 4]));
    const csr = denseToCsr(t);
    const coo = sparseToCoo(csr);
    expect(coo.format).toBe('COO');
    expect(unpackData(coo.valuesTensor!)).toEqual([1, 4]);
    expect(unpackData(coo.indicesTensor!)).toEqual([0, 3]);

    expect(sparseToCoo(coo)).toBe(coo);

    const dense = sparseToDense(csr);
    expect(unpackData(dense)).toEqual([1, 0, 0, 4]);

    const emptySparse = new SparseTensor('e', [2, 2]);
    expect(unpackData(sparseToDense(emptySparse))).toEqual([]);
  });
});
