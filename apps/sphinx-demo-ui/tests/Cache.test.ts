// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { Cache } from '../src/core/Cache.js';

describe('Cache', () => {
  it('should store and evict', () => {
    vi.useFakeTimers();
    const cache = new Cache<number>(1000);

    cache.set('a', 1);
    expect(cache.get('a')).toBe(1);

    vi.advanceTimersByTime(2000);
    expect(cache.get('a')).toBeUndefined();

    vi.useRealTimers();
  });
});
