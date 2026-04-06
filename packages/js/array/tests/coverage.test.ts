import { describe, expect, test } from 'vitest';
import * as np from '../src/index.js';

describe('onnx9000.array coverage', () => {
  const eagerA = np.array([1, 2, 3]);
  const lazyA = np.Input('x', [3], 'float32');

  const ops = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'power',
    'mod',
    'absolute',
    'negative',
    'sign',
    'exp',
    'log',
    'sqrt',
    'square',
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'matmul',
    'equal',
    'less',
    'greater',
    'less_equal',
    'greater_equal',
    'logical_and',
    'logical_or',
    'logical_not',
    'logical_xor',
    'isnan',
    'isinf',
    'where',
    'sum',
    'prod',
    'mean',
    'min',
    'max',
    'argmin',
    'argmax',
    'reshape',
    'squeeze',
    'expand_dims',
    'concatenate',
    'split',
    'tile',
    'pad',
    'transpose',
    'take',
    'gather',
    'sort',
    'argsort',
    'nonzero',
    'zeros',
    'ones',
    'empty',
    'full',
    'eye',
    'identity',
    'arange',
    'linspace',
    'log10',
    'log2',
    'cbrt',
    'reciprocal',
    'deg2rad',
    'rad2deg',
    'dot',
    'vdot',
    'inner',
    'outer',
    'tensordot',
    'einsum',
    'swapaxes',
    'trace',
    'ptp',
    'all',
    'any',
    'cumsum',
    'cumprod',
    'ravel',
    'broadcast_to',
    'stack',
    'vstack',
    'hstack',
    'dstack',
    'array_split',
    'repeat',
    'not_equal',
    'allclose',
    'isclose',
    'extract',
    'take_along_axis',
    'put',
    'put_along_axis',
    'nan_to_num',
    'clip',
    'around',
    'fix',
    'i0',
    'sinc',
    'save',
    'load',
    'vectorize',
    'meshgrid',
    'mgrid',
    'einsum_path',
    'polyfit',
    'histogram',
    'digitize',
    'export_model',
    'compile',
    'set_device',
    'set_log_level',
    'set_opset',
    'set_num_threads',
  ];

  test('all functional operations', () => {
    for (const opName of ops) {
      const op = (np as Object)[opName];
      if (typeof op !== 'function') continue;

      // Eager
      np.lazy_mode(false);
      const resEager = op(eagerA, eagerA);
      expect(resEager).toBeInstanceOf(np.EagerTensor);

      // Lazy
      np.lazy_mode(true);
      const resLazy = op(lazyA, lazyA);
      expect(resLazy).toBeInstanceOf(np.LazyTensor);
    }
  });

  test('nn operations', () => {
    const nnOps = Object.keys(np.nn);
    for (const opName of nnOps) {
      const op = (np.nn as Object)[opName];

      // Eager
      np.lazy_mode(false);
      const resEager = op(eagerA, eagerA);
      expect(resEager).toBeInstanceOf(np.EagerTensor);

      // Lazy
      np.lazy_mode(true);
      const resLazy = op(lazyA, lazyA);
      expect(resLazy).toBeInstanceOf(np.LazyTensor);
    }
  });

  test('linalg operations', () => {
    const linalgOps = Object.keys(np.linalg);
    for (const opName of linalgOps) {
      const op = (np.linalg as Object)[opName];

      // Eager
      np.lazy_mode(false);
      const resEager = op(eagerA, eagerA);
      expect(resEager).toBeInstanceOf(np.EagerTensor);

      // Lazy
      np.lazy_mode(true);
      const resLazy = op(lazyA, lazyA);
      expect(resLazy).toBeInstanceOf(np.LazyTensor);
    }
  });

  test('char operations', () => {
    const charOps = Object.keys(np.char);
    for (const opName of charOps) {
      const op = (np.char as Object)[opName];

      // Eager
      np.lazy_mode(false);
      const resEager = op('a', 'b');
      expect(resEager).toBeInstanceOf(np.EagerTensor);

      // Lazy
      np.lazy_mode(true);
      const resLazy = op('a', 'b');
      expect(resLazy).toBeInstanceOf(np.LazyTensor);
    }
  });

  test('random operations', () => {
    const randomOps = ['rand', 'randn', 'randint', 'uniform', 'normal'];
    for (const opName of randomOps) {
      const op = (np.random as Object)[opName];

      // Eager
      np.lazy_mode(false);
      const resEager = op([2, 2]);
      expect(resEager).toBeInstanceOf(np.EagerTensor);

      // Lazy
      np.lazy_mode(true);
      const resLazy = op([2, 2]);
      expect(resLazy).toBeInstanceOf(np.LazyTensor);
    }
    np.random.seed(42);
  });

  test('EagerTensor methods', () => {
    np.lazy_mode(false);
    const t = np.array([1, 2, 3]);
    expect(t.ndim).toBe(1);

    const t2 = np.array(new Uint8Array([1, 2]));
    expect(t2.ndim).toBe(1);
    expect(t2.shape[0]).toBe(2);

    const t3 = np.array(null);
    expect(t3.ndim).toBe(1);
    expect(t3.shape[0]).toBe(0);

    expect(t.numpy()).toEqual([1, 2, 3]);
    expect(t.data_val()).toEqual([1, 2, 3]);
    expect(t.evaluate()).toBe(t);
    expect(t.cpu()).toBe(t);
    expect(t.gpu()).toBe(t);
    expect(t.quantize_dynamic()).toBe(t);
    expect(t.T).toBeInstanceOf(np.EagerTensor);

    // Call some methods
    expect(t.add(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.subtract(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.multiply(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.divide(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.power(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.mod(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.absolute()).toBeInstanceOf(np.EagerTensor);
    expect(t.negative()).toBeInstanceOf(np.EagerTensor);
    expect(t.sign()).toBeInstanceOf(np.EagerTensor);
    expect(t.exp()).toBeInstanceOf(np.EagerTensor);
    expect(t.log()).toBeInstanceOf(np.EagerTensor);
    expect(t.sqrt()).toBeInstanceOf(np.EagerTensor);
    expect(t.square(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.sin()).toBeInstanceOf(np.EagerTensor);
    expect(t.cos()).toBeInstanceOf(np.EagerTensor);
    expect(t.tan()).toBeInstanceOf(np.EagerTensor);
    expect(t.arcsin()).toBeInstanceOf(np.EagerTensor);
    expect(t.arccos()).toBeInstanceOf(np.EagerTensor);
    expect(t.arctan()).toBeInstanceOf(np.EagerTensor);
    expect(t.sinh()).toBeInstanceOf(np.EagerTensor);
    expect(t.cosh()).toBeInstanceOf(np.EagerTensor);
    expect(t.tanh()).toBeInstanceOf(np.EagerTensor);
    expect(t.arcsinh()).toBeInstanceOf(np.EagerTensor);
    expect(t.arccosh()).toBeInstanceOf(np.EagerTensor);
    expect(t.arctanh()).toBeInstanceOf(np.EagerTensor);
    expect(t.matmul(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.equal(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.less(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.greater(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.less_equal(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.greater_equal(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.logical_and(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.logical_or(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.logical_not()).toBeInstanceOf(np.EagerTensor);
    expect(t.logical_xor(2)).toBeInstanceOf(np.EagerTensor);
    expect(t.isnan()).toBeInstanceOf(np.EagerTensor);
    expect(t.isinf()).toBeInstanceOf(np.EagerTensor);

    t.dispose();
    expect(t.data).toBeNull();
  });

  test('Errors', () => {
    expect(new np.BroadcastError()).toBeInstanceOf(Error);
    expect(new np.TypeMismatchError()).toBeInstanceOf(Error);
  });
});
