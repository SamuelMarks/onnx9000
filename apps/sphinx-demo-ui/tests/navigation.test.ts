// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { initNavigation } from '../docs/assets/navigation.js';

describe('navigation', () => {
  it('should run', () => {
    initNavigation();
    expect(true).toBe(true);
  });
});
