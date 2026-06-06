// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { initMain } from '../docs/assets/main.js';

describe('main', () => {
  it('should run', () => {
    initMain();
    expect(true).toBe(true);
  });
});
