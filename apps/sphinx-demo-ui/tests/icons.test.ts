// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { initIcons } from '../docs/assets/icons.js';

describe('icons', () => {
  it('should run', () => {
    initIcons();
    expect(true).toBe(true);
  });
});
