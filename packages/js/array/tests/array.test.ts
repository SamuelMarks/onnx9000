import { describe, expect, test } from 'vitest';
import * as np from '../src/index.js';

describe('onnx9000.array', () => {
  test('core instantiation', () => {
    const a = np.array([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.data).toEqual([1, 2, 3]);
    expect(a.numpy()).toEqual([1, 2, 3]);
    expect(a.data_val()).toEqual([1, 2, 3]);
    expect(a.ndim).toBe(1);

    a.dispose();
    expect(a.data).toBeNull();
  });

  test('lazy context', () => {
    np.lazy_mode(true);
    const x = np.Input('x', [1, 2], 'float32');
    expect(x).toBeInstanceOf(np.LazyTensor);
    expect((x as Object).opType).toBe('Input');

    const y = np.add(x, 2);
    expect(y).toBeInstanceOf(np.LazyTensor);
    expect((y as Object).opType).toBe('Add');

    const z = np.matmul(x, y);
    expect(z).toBeInstanceOf(np.LazyTensor);
    expect((z as Object).opType).toBe('MatMul');
    np.lazy_mode(false);
  });

  test('math operations eager', () => {
    const a = np.array([1, 2]);
    const b = np.add(a, 2);
    expect(b).toBeInstanceOf(np.EagerTensor);

    expect(np.sin(a)).toBeInstanceOf(np.EagerTensor);
    expect(np.exp(a)).toBeInstanceOf(np.EagerTensor);
    expect(np.reshape(a, [2, 1])).toBeInstanceOf(np.EagerTensor);
  });

  test('nn operations', () => {
    const x = np.array([1, 2]);
    np.lazy_mode(true);
    const z = np.nn.relu(x);
    expect(z).toBeInstanceOf(np.LazyTensor);
    expect((z as Object).opType).toBe('Relu');
    np.lazy_mode(false);
  });

  test('linalg operations', () => {
    const x = np.array([1, 2]);
    np.lazy_mode(true);
    const z = np.linalg.det(x);
    expect(z).toBeInstanceOf(np.LazyTensor);
    expect((z as Object).opType).toBe('Det');
    np.lazy_mode(false);
  });

  test('random operations', () => {
    np.lazy_mode(true);
    const z = np.random.randn([2, 2]);
    expect(z).toBeInstanceOf(np.LazyTensor);
    expect((z as Object).opType).toBe('RandomNormal');
    np.lazy_mode(false);
  });
});
