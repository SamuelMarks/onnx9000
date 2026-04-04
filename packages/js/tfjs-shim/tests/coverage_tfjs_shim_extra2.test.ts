import { describe, it, expect } from 'vitest';
import * as tf from '../src/index';

describe('Coverage tfjs-shim Extra 2', () => {
  it('scalar, buffer, clone', () => {
    expect(tf.scalar(5).shape).toEqual([]);
    expect(tf.buffer([2]).shape).toEqual([2]);
    expect(tf.buffer([2], 'float32', [1, 2]).dataSync()[1]).toBe(2);
    expect(tf.clone(tf.scalar(1)).dataSync()[0]).toBe(1);
  });

  it('step, addN', () => {
    expect(tf.step(tf.tensor([1, -1]), 0.5).dataSync()[1]).toBe(0.5);
    expect(tf.addN([tf.tensor([1]), tf.tensor([2]), tf.tensor([3])]).dataSync()[0]).toBe(6);
    expect(() => tf.addN([])).toThrow();
  });

  it('max, mean, prod', () => {
    const t = tf.tensor([1, 2, 3]);
    expect(tf.max(t).dataSync()[0]).toBe(3);
    expect(tf.max(t, undefined, true).shape).toEqual([1]);

    expect(tf.mean(t).dataSync()[0]).toBe(2);
    expect(tf.mean(t, undefined, true).shape).toEqual([1]);

    expect(tf.prod(t).dataSync()[0]).toBe(6);
    expect(tf.prod(t, undefined, true).shape).toEqual([1]);
  });

  it('conv2d and depthwise NCHW', () => {
    const x = tf.tensor([1, 2, 3, 4], [1, 1, 2, 2]); // NCHW
    const f = tf.tensor([1], [1, 1, 1, 1]);
    const c1 = tf.conv2d(x, f, 1, 'valid', 'NCHW');
    expect(c1.shape).toEqual([1, 1, 2, 2]);

    const f2 = tf.tensor([1], [1, 1, 1, 1]); // depthwise
    const c2 = tf.depthwiseConv2d(x, f2, 1, 'valid', 'NCHW');
    expect(c2.shape).toEqual([1, 1, 2, 2]);
  });

  it('boolean elementwise and dispose', () => {
    const t = tf.add(true, false);
    expect(t.dataSync()[0]).toBe(1);
    tf.dispose([t]);

    const t2 = tf.tensor([1]);
    tf.dispose({ t: t2 });

    const t3 = tf.tensor([1]);
    const t4 = tf.tensor([1]);
    tf.dispose(t3);
    tf.dispose(t3); // double dispose shouldn't crash
    tf.keep(t4);
  });
});
