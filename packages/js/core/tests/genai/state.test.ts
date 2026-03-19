import { test, expect } from 'vitest';
import { Tensor } from '../../src/ir/tensor';
import { GroupedQueryAttentionCache, MultiQueryAttentionCache } from '../../src/genai/state';

test('GroupedQueryAttentionCache invalid heads', () => {
  const kvc = new GroupedQueryAttentionCache(1, 32);
  const k = new Tensor('k', 'float32', [1, 2, 10, 32]);
  const v = new Tensor('v', 'float32', [1, 2, 10, 32]);
  expect(() => kvc.update(k, v, 0)).toThrowError(/Expected 1 KV heads/);
});

test('MultiQueryAttentionCache invalid heads', () => {
  const kvc = new MultiQueryAttentionCache(32);
  const k = new Tensor('k', 'float32', [1, 2, 10, 32]);
  const v = new Tensor('v', 'float32', [1, 2, 10, 32]);
  expect(() => kvc.update(k, v, 0)).toThrowError(/Expected 1 KV heads/);
});
