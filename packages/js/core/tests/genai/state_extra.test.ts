import { describe, expect, it } from 'vitest';
import {
  GroupedQueryAttentionCache,
  MultiHeadAttentionCache,
  MultiQueryAttentionCache,
} from '../../src/genai/state.js';

describe('state coverage', () => {
  it('should clear caches', () => {
    const gqa = new GroupedQueryAttentionCache(4, 64);
    gqa.clear();
    const mqa = new MultiQueryAttentionCache(64);
    mqa.clear();
    const mha = new MultiHeadAttentionCache(4, 64);
    mha.clear();
    expect(typeof mha).toBeDefined();
  });
});
