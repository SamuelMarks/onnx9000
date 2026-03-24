import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Cache } from '../../src/core/Cache';

describe('Cache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should store and retrieve items', () => {
    const cache = new Cache<string>(10000);
    cache.set('key1', 'value1');
    expect(cache.get('key1')).toBe('value1');
  });

  it('should return undefined for missing items', () => {
    const cache = new Cache<string>();
    expect(cache.get('missing')).toBeUndefined();
  });

  it('should evict items when they expire on get', () => {
    const cache = new Cache<string>(100);
    cache.set('key1', 'value1');

    // Fast-forward past TTL
    vi.advanceTimersByTime(150);

    expect(cache.get('key1')).toBeUndefined();
  });

  it('should respect custom TTLs overriding the default', () => {
    const cache = new Cache<string>(1000); // 1 sec default

    cache.set('short', 'val', 100); // 100ms
    cache.set('long', 'val', 5000); // 5s

    vi.advanceTimersByTime(200);

    expect(cache.get('short')).toBeUndefined(); // Expired
    expect(cache.get('long')).toBe('val'); // Still alive
  });

  it('should manually clear all items', () => {
    const cache = new Cache<number>();
    cache.set('a', 1);
    cache.set('b', 2);

    cache.clear();

    expect(cache.get('a')).toBeUndefined();
    expect(cache.get('b')).toBeUndefined();
  });

  it('should passively evict expired items without a get call', () => {
    const cache = new Cache<string>(100);

    cache.set('a', '1');
    cache.set('b', '2', 500); // Will survive

    vi.advanceTimersByTime(200);

    // Call evictExpired manually (in a real app, this might be on an interval)
    cache.evictExpired();

    // Since 'a' is evicted, it's completely gone from internal map
    // We can test this by checking internal size via an any cast
    const internalMap = (cache as any).store as Map<string, any>;

    expect(internalMap.has('a')).toBe(false);
    expect(internalMap.has('b')).toBe(true);
  });
});
