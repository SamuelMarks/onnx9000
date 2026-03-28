/* eslint-disable */
import { describe, it, expect } from 'vitest';
import * as tf from '../src/index';

describe('tfjs-shim', () => {
  it('should create tensors and run basic math', async () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = tf.add(a, b);
    const data = await c.data();
    expect(data[0]).toBe(5);
    expect(data[1]).toBe(7);
    expect(data[2]).toBe(9);
  });

  it('should support tidy', () => {
    const startNum = tf.memory().numTensors;
    tf.tidy(() => {
      const a = tf.tensor([1, 2]);
      const b = tf.tensor([3, 4]);
      tf.add(a, b);
    });
    expect(tf.memory().numTensors).toBe(startNum);
  });

  it('should support layers', () => {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 1 }));
    expect(m.layers.length).toBe(1);
  });
});
