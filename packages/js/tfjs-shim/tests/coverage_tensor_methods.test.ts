import { describe, it, expect } from 'vitest';
import tf from '../src/index.js';

describe('Tensor methods coverage', () => {
  it('covers array() and arraySync()', async () => {
    const t0 = tf.scalar(42);
    expect(t0.arraySync()).toBe(42);
    expect(await t0.array()).toBe(42);

    const t1 = tf.tensor1d([1, 2, 3]);
    expect(t1.arraySync()).toEqual([1, 2, 3]);
    expect(await t1.array()).toEqual([1, 2, 3]);

    const t2 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    expect(t2.arraySync()).toEqual([1, 2, 3, 4]);
    expect(await t2.array()).toEqual([1, 2, 3, 4]);
  });
});
