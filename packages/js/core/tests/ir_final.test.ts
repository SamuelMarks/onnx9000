import { describe, it, expect } from 'vitest';
import { Tensor, SparseTensor } from '../src/ir/tensor.js';

describe('IR Tensor Gaps', () => {
  it('formatData more coverage', () => {
    const data = new Float32Array([1, 2, 3]);
    const t = new Tensor('t', [3], 'float32', true, false, data);
    expect(t.formatData(2)).toContain('elements]');

    const t2 = new Tensor('t2', [2], 'int64', true, false, new BigInt64Array([1n, 2n]));
    expect(t2.formatData()).toBe('[1, 2]');

    const t3 = new Tensor(
      't3',
      [1],
      'float16',
      true,
      false,
      new Uint8Array(new Uint16Array([0x3c00]).buffer),
    ); // 1.0
    expect(t3.formatData()).toBe('[1]');

    const t4 = new Tensor(
      't4',
      [1],
      'bfloat16',
      true,
      false,
      new Uint8Array(new Uint16Array([0x3f80]).buffer),
    ); // 1.0
    expect(t4.formatData()).toBe('[1]');

    const t5 = new Tensor('t5', [1], 'int8', true, false, new Int8Array([5]));
    expect(t5.formatData()).toBe('[5]');
  });

  it('copy more coverage', () => {
    const t = new Tensor('t', [2], 'float32', true, true, new Float32Array([1, 2]));
    t.externalData = { location: 'loc', offset: 0, length: 8 };
    const c = t.copy();
    expect(c.name).toBe('t');
    expect(c.externalData?.location).toBe('loc');

    const st = new SparseTensor('st', [2, 2], 'COO', t, t);
    const sc = st.copy() as SparseTensor;
    expect(sc.format).toBe('COO');
    expect(sc.valuesTensor?.name).toBe('t');
  });
});
