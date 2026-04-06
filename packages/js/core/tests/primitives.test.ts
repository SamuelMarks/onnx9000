import { test, expect } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import {
  BaseNorm,
  BatchNormalization,
  LayerNormalization,
  RMSNorm,
  GroupNorm,
  InstanceNorm,
} from '../src/primitives.js';

class MockBaseNorm extends BaseNorm {
  call(...args: Object[]): Tensor {
    return new Tensor('mock', [], 1);
  }
}

test('BaseNorm abstract', () => {
  const norm = new MockBaseNorm();
  expect(norm.epsilon).toBe(1e-5);
  expect(norm.call().name).toBe('mock');
});

test('BatchNormalization', () => {
  const norm = new BatchNormalization(3);
  const x = new Tensor('x', [1, 3, 2, 2], 1);
  const scale = new Tensor('scale', [3], 1);
  const b = new Tensor('b', [3], 1);
  const mean = new Tensor('mean', [3], 1);
  const varT = new Tensor('var', [3], 1);

  const out = norm.call(x, scale, b, mean, varT);
  expect(out.name).toBe('BatchNormalization_out');
});

test('LayerNormalization', () => {
  const norm = new LayerNormalization([2, 2]);
  const x = new Tensor('x', [1, 3, 2, 2], 1);
  const scale = new Tensor('scale', [2, 2], 1);
  const b = new Tensor('b', [2, 2], 1);

  const out = norm.call(x, scale, b);
  expect(out.name).toBe('LayerNormalization_out');
  expect(norm.axis).toBe(-2);
});

test('LayerNormalization without bias', () => {
  const norm = new LayerNormalization([2, 2]);
  const x = new Tensor('x', [1, 3, 2, 2], 1);
  const scale = new Tensor('scale', [2, 2], 1);

  const out = norm.call(x, scale);
  expect(out.name).toBe('LayerNormalization_out');
});

test('RMSNorm', () => {
  const norm = new RMSNorm([2, 2]);
  const x = new Tensor('x', [1, 3, 2, 2], 1);
  const scale = new Tensor('scale', [2, 2], 1);

  const out = norm.call(x, scale);
  expect(out.name).toBe('RMSNormalization_out');
});

test('GroupNorm', () => {
  const norm = new GroupNorm(2, 4);
  const x = new Tensor('x', [1, 4, 2, 2], 1);
  const scale = new Tensor('scale', [4], 1);
  const b = new Tensor('b', [4], 1);

  const out = norm.call(x, scale, b);
  expect(out.name).toBe('GroupNormalization_out');
});

test('InstanceNorm', () => {
  const norm = new InstanceNorm(4);
  const x = new Tensor('x', [1, 4, 2, 2], 1);
  const scale = new Tensor('scale', [4], 1);
  const b = new Tensor('b', [4], 1);

  const out = norm.call(x, scale, b);
  expect(out.name).toBe('InstanceNormalization_out');
});

import {
  BaseActivation,
  Relu,
  Sigmoid,
  Tanh,
  LeakyRelu,
  Gelu,
  Silu,
  Swish,
  Mish,
} from '../src/primitives.js';

class MockBaseAct extends BaseActivation {
  call(x: Tensor): Tensor {
    return new Tensor('mock', [], 1);
  }
}

test('BaseActivation abstract', () => {
  const act = new MockBaseAct();
  expect(act.call(new Tensor('x', [], 1)).name).toBe('mock');
  expect(act.generateLUT().name).toBe('Constant_out');
});

test('Activations', () => {
  const x = new Tensor('x', [1], 1);
  expect(new Relu().call(x).name).toBe('Relu_out');
  expect(new Sigmoid().call(x).name).toBe('Sigmoid_out');
  expect(new Tanh().call(x).name).toBe('Tanh_out');
  expect(new LeakyRelu(0.01).call(x).name).toBe('LeakyRelu_out');
  expect(new Gelu().call(x).name).toBe('Gelu_out');
  expect(new Silu().call(x).name).toBe('Swish_out');
  expect(new Swish().call(x).name).toBe('Swish_out');
  expect(new Mish().call(x).name).toBe('Mish_out');
});

import { ConvFamily, ConvND, DepthwiseConv, MatMul, Gemm } from '../src/primitives.js';

test('ConvFamily abstract', () => {
  const fam = new ConvFamily(1, 1, 3);
  expect(() => fam.call(new Tensor('x', [], 1), new Tensor('w', [], 1))).toThrow();
});

test('ConvND', () => {
  const conv = new ConvND(2, 3, 4, 3);
  const out = conv.call(new Tensor('x', [], 1), new Tensor('w', [], 1));
  expect(out.name).toBe('Conv_out');
});

test('DepthwiseConv', () => {
  const conv = new DepthwiseConv(2, 3, 3);
  const out = conv.call(new Tensor('x', [], 1), new Tensor('w', [], 1));
  expect(out.name).toBe('Conv_out');
  expect(conv.groups).toBe(3);
});

test('MatMul / Gemm', () => {
  const x = new Tensor('x', [], 1);
  const y = new Tensor('y', [], 1);
  expect(new MatMul().call(x, y).name).toBe('MatMul_out');
  expect(new Gemm().call(x, y).name).toBe('Gemm_out');
});

import {
  MultiHeadAttention,
  FlashAttention,
  GroupedQueryAttention,
  RoPE,
  AlibiBias,
} from '../src/primitives.js';

test('Attention', () => {
  const q = new Tensor('q', [1, 8, 32, 64], 1);
  const k = new Tensor('k', [1, 8, 32, 64], 1);
  const v = new Tensor('v', [1, 8, 32, 64], 1);

  expect(new MultiHeadAttention(8).call(q, k, v).name).toBe('Attention_out');
  expect(new FlashAttention(8).call(q, k, v).name).toBe('FlashAttention_out');
  expect(new GroupedQueryAttention(8, 2).call(q, k, v).name).toBe('GroupedQueryAttention_out');
});

test('RoPE / AlibiBias', () => {
  const x = new Tensor('x', [1, 32, 64], 1);
  const pos = new Tensor('pos', [1, 32], 1);
  expect(new RoPE(64).call(x, pos).name).toBe('RoPE_out');

  const mask = new Tensor('mask', [1, 8, 32, 32], 1);
  expect(new AlibiBias(8).call(mask).name).toBe('AlibiBias_out');
});
