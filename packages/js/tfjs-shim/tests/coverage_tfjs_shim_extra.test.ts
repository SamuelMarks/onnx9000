import { describe, it, expect } from 'vitest';
import * as tf from '../src/index';

describe('Coverage tfjs-shim Extra', () => {
  it('data() and dataSync()', async () => {
    const t1 = tf.tensor([1, 2], [2], 'int32');
    expect(await t1.data()).toBeInstanceOf(Int32Array);
    expect(t1.dataSync()).toBeInstanceOf(Int32Array);

    const t2 = tf.tensor([true, false], [2], 'bool');
    expect(await t2.data()).toBeInstanceOf(Uint8Array);
    expect(t2.dataSync()).toBeInstanceOf(Uint8Array);

    const t3 = tf.tensor(['a', 'b'], [2], 'string');
    expect(await t3.data()).toBeInstanceOf(Array);
    expect(t3.dataSync()).toBeInstanceOf(Array);
  });

  it('expandDims, reshapeAs, cast, squeeze', () => {
    const t = tf.tensor([1, 2], [2]);
    expect(tf.expandDims(t, 0).shape).toEqual([1, 2]);
    expect(tf.reshape(t, [1, 2]).shape).toEqual([1, 2]);

    // Use t.cast and t.squeeze to hit class methods
    expect(t.cast('int32').dtype).toBe('int32');
    expect(tf.tensor([1], [1, 1]).squeeze().shape).toEqual([]);

    // Also call standalone cast and squeeze
    expect(tf.cast(t, 'int32').dtype).toBe('int32');
    expect(tf.squeeze(tf.tensor([1], [1, 1])).shape).toEqual([]);
  });

  it('pad', () => {
    const t = tf.tensor([1, 2], [2]);
    expect(tf.pad(t, [[1, 1]]).shape).toEqual([4]);
  });

  it('tensor creation 3d, 4d, 5d, 6d', () => {
    expect(tf.tensor3d([[[1]]], [1, 1, 1]).shape).toEqual([1, 1, 1]);
    expect(tf.tensor4d([[[[1]]]], [1, 1, 1, 1]).shape).toEqual([1, 1, 1, 1]);
    expect(tf.tensor5d([[[[[1]]]]], [1, 1, 1, 1, 1]).shape).toEqual([1, 1, 1, 1, 1]);
    expect(tf.tensor6d([[[[[[1]]]]]], [1, 1, 1, 1, 1, 1]).shape).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it('tidy scope and keep and dispose', () => {
    const res = tf.tidy(() => {
      const t1 = tf.tensor([1]);
      const t2 = tf.tensor([2]);
      tf.keep(t1);
      return [t2, { a: tf.tensor([3]) }];
    });
    expect(res).toBeDefined();

    const arr = res as Object[];
    tf.dispose([arr[0]]);
    tf.dispose(arr[1]);
  });

  it('conv1d', () => {
    const x = tf.tensor([1, 2, 3], [1, 3, 1]);
    const f = tf.tensor([1], [1, 1, 1]);

    // valid
    const c1 = tf.conv1d(x, f, 1, 'valid');
    expect(c1.shape).toEqual([1, 3, 1]);

    // NCW
    const x2 = tf.tensor([1, 2, 3], [1, 1, 3]);
    const c2 = tf.conv1d(x2, f, 1, 'same', 'NCW');
    expect(c2.shape).toEqual([1, 1, 3]);

    // pad num
    const c3 = tf.conv1d(x, f, 1, 1);
    expect(c3.shape).toEqual([1, 5, 1]);
  });

  it('conv3d', () => {
    const x = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const f = tf.tensor([1], [1, 1, 1, 1, 1]);

    // NDHWC
    const c1 = tf.conv3d(x, f, [1, 1, 1], 'valid');
    expect(c1.shape).toEqual([1, 2, 2, 2, 1]);

    // same
    const c2 = tf.conv3d(x, f, 1, 'same');
    expect(c2.shape).toEqual([1, 2, 2, 2, 1]);

    // NCDHW
    const x2 = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2]);
    const c3 = tf.conv3d(x2, f, 1, 'valid', 'NCDHW', [1, 1, 1]);
    expect(c3.shape).toEqual([1, 1, 2, 2, 2]);
  });

  it('norm', () => {
    const t = tf.tensor([-3, 4], [2]);
    expect(tf.norm(t, 1).dataSync()[0]).toBe(7);
    expect(tf.norm(t, 2).dataSync()[0]).toBe(5);
    expect(tf.norm(t, 'euclidean').dataSync()[0]).toBe(5);
    expect(tf.norm(t, Infinity).dataSync()[0]).toBe(4);
    expect(tf.norm(t, 'inf').dataSync()[0]).toBe(4);
  });
});
