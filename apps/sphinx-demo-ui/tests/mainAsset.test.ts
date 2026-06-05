// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { initMain } from '../docs/assets/main.js';

describe('main', () => {
  it('should run', () => {
    initMain();
    expect(true).toBe(true);
  });
});
