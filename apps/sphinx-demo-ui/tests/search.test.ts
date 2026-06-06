// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { initSearch } from '../docs/assets/search.js';

describe('search', () => {
  it('should run', () => {
    initSearch();
    expect(true).toBe(true);
  });
});
