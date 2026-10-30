import { describe, expect, it } from 'vitest';
import { initMain } from '../docs/assets/main.js';

describe('main', () => {
  it('should run initMain and return undefined', () => {
    expect(typeof initMain).toBe('function');
    expect(initMain()).toBeUndefined();
  });
});
