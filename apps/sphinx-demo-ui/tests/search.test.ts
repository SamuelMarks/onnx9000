import { describe, expect, it } from 'vitest';
import { initSearch } from '../docs/assets/search.js';

describe('search', () => {
  it('should run initSearch and return undefined', () => {
    expect(typeof initSearch).toBe('function');
    expect(initSearch()).toBeUndefined();
  });
});
