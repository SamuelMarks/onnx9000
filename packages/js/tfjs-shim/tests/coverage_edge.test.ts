import { describe, it, expect } from 'vitest';
import tf from '../src/index.js';
import { tensor } from '../src/index.js';

describe('Edge cases coverage', () => {
  it('covers backend functions', async () => {
    await tf.setBackend('webgl');
    expect(tf.getBackend()).toBe('webgl');
  });

  it('covers data() and dataSync() with unknown dtype', async () => {
    const t = tf.tensor([1], [1], 'complex64' as any);
    expect(t.dataSync()).toBeInstanceOf(Float32Array);
    expect(await t.data()).toBeInstanceOf(Float32Array);
  });

  it('covers tensor() with primitive value', () => {
    const t = tensor(42);
    expect(t.shape).toEqual([1]);
  });

  it('covers tidy() with non-tensor return and tensor disposal', () => {
    let tToDispose: tf.Tensor;
    const res = tf.tidy(() => {
      tToDispose = tf.tensor(1);
      return null as any;
    });
    expect(res).toBeNull();
    expect(tToDispose!.isDisposed).toBe(true);
  });

  it('covers invalid inputs to elementwise op', () => {
    expect(() => tf.add(undefined as any, 1)).toThrow(/Invalid inputs to add/);
  });
});
