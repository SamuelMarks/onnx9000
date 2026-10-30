import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { AddOp, MulOp, SubOp } from '../src/ops/index.js';

describe('ops', () => {
  it('should run operations', () => {
    const t1 = new Tensor('a', [1], 'float32', false, true, new Float32Array([1]));
    const t2 = new Tensor('b', [1], 'float32', false, true, new Float32Array([2]));

    const add = new AddOp();
    const res = add.execute([t1, t2], {});
    expect((res[0]?.data as Float32Array | undefined)?.[0]).toBe(3);

    const sub = new SubOp();
    const res2 = sub.execute([t1, t2], {});
    expect((res2[0]?.data as Float32Array | undefined)?.[0]).toBe(-1);

    const mul = new MulOp();
    const res3 = mul.execute([t1, t2], {});
    expect((res3[0]?.data as Float32Array | undefined)?.[0]).toBe(2);
  });
});
