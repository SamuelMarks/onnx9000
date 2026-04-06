import { describe, it, expect } from 'vitest';
import * as idx from '../src/index';
import { compileGGUF } from '../src/compiler';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('Coverage onnx2gguf', () => {
  it('index', () => {
    expect(idx).toBeDefined();
  });

  it('Compiler docString and mixed types', async () => {
    const g = new Graph('test');
    (g as Object).docString = 'hello';

    const overrides = {
      'general.bool_val': true,
      'general.float_val': 1.5,
      'general.int_val': 42,

      // non-general
      'arch.bool_val': true,
      'arch.float_val': 1.5,
      'arch.int_val': 42,
      'arch.arr_float': [1.5, 2.5],
      'arch.arr_str': ['a', 'b'],
      'arch.arr_int': [1, 2],
      'arch.arr_bool': [true, false], // tests the INT32 fallback
    };

    // Add a non-float32/float16 tensor to trigger fallback
    g.tensors['t1'] = new Tensor('t1', [1], 'int32', false, true, new Int32Array([1]));
    g.initializers.push('t1');

    // Override arch to 'gemma' to get float values in archMeta
    const out = await compileGGUF(g, overrides, 'gemma');
    expect(out).toBeDefined();
  });
});
