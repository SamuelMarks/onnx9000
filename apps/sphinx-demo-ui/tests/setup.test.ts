// @ts-nocheck
import { describe, it, expect } from 'vitest';

describe('setup', () => {
  it('should be valid', async () => {
    await import('../tests/setup.js');
    expect(global.ResizeObserver).toBeDefined();
  });
});
