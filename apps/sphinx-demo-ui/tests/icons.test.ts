// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { initIcons } from '../docs/assets/icons.js';

describe('icons', () => {
  it('should run', () => {
    initIcons();
    expect(true).toBe(true);
  });
});
