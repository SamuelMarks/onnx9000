// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { initSearch } from '../docs/assets/search.js';

describe('search', () => {
  it('should run', () => {
    initSearch();
    expect(true).toBe(true);
  });
});
