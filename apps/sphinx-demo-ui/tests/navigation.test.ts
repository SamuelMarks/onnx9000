// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { initNavigation } from '../docs/assets/navigation.js';

describe('navigation', () => {
  it('should run', () => {
    initNavigation();
    expect(true).toBe(true);
  });
});
