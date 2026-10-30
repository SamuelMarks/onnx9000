import { describe, expect, it } from 'vitest';
import { initIcons } from '../docs/assets/icons.js';

describe('icons', () => {
  it('should run initIcons and return undefined', () => {
    expect(typeof initIcons).toBe('function');
    expect(initIcons()).toBeUndefined();
  });
});
