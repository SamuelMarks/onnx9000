import { describe, it, expect } from 'vitest';
import { ArrayAPI } from '../src/array/index';

describe('ArrayAPI', () => {
  it('softmax, log_softmax, sigmoid', () => {
    const input = [1, 2, 3];
    expect(ArrayAPI.softmax(input)).toBeDefined();
    expect(ArrayAPI.log_softmax(input)).toBeDefined();
    expect(ArrayAPI.sigmoid(input)).toBeDefined();
  });

  it('add, get_top_k, cosine_similarity, dot_product', () => {
    const a = [1, 2, 3];
    const b = [4, 5, 6];
    expect(ArrayAPI.add(a, b)).toEqual([5, 7, 9]);

    const topK = ArrayAPI.get_top_k([1, 10, 5], 2);
    expect(topK.values).toEqual([10, 5]);
    expect(topK.indices).toEqual([1, 2]);

    expect(ArrayAPI.cosine_similarity(a, b)).toBeDefined();
    expect(ArrayAPI.dot_product(a, b)).toEqual(32);
  });

  it('tensor manipulation', () => {
    const t = [1, 2, 3, 4] as any;
    expect(ArrayAPI.view(t, [2, 2])).toBe(t);
    expect(ArrayAPI.reshape(t, [2, 2])).toBe(t);
    expect(ArrayAPI.transpose(t, [1, 0])).toBe(t);
    expect(ArrayAPI._maybeDispatchWasm(t)).toBe(false);
    expect(ArrayAPI._maybeDispatchWasm(new Array(10001).fill(0))).toBe(true);
  });

  it('type conversions', () => {
    const t = [1, 2, 3];
    const f32 = ArrayAPI.toFloat32Array(t);
    expect(f32).toBeInstanceOf(Float32Array);
    expect(ArrayAPI.fromFloat32Array(f32)).toEqual(t);
    expect(ArrayAPI.toJSON(t)).toBe(t);
    expect(ArrayAPI.toJSON('not array')).toEqual([]);
    expect(ArrayAPI.fromJSON(t)).toBe(t);
  });

  it('slice and stride', () => {
    const t = [1, 2, 3, 4];
    expect(ArrayAPI.slice(t, 1, 3)).toEqual([2, 3]);
    expect(ArrayAPI.getStrided(t, 2)).toEqual([1, 3]);
    expect(ArrayAPI.getStrided('not array', 2)).toEqual('not array');
  });

  it('erf', () => {
    expect(ArrayAPI.erf(0)).toBeCloseTo(0);
    expect(ArrayAPI.erf(1)).toBeGreaterThan(0.8);
    expect(ArrayAPI.erf(-1)).toBeLessThan(-0.8);
  });
});
