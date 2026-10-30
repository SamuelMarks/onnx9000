import { describe, expect, it } from 'vitest';
import { initNavigation } from '../docs/assets/navigation.js';

describe('navigation', () => {
  it('should run initNavigation and return undefined', () => {
    expect(typeof initNavigation).toBe('function');
    expect(initNavigation()).toBeUndefined();
  });
});
