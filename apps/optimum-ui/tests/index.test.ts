import { describe, it, expect } from 'vitest';

describe('optimum-ui index', () => {
  it('should be valid', async () => {
    const mod = await import('../src/index.js');
    expect(mod).toBeDefined();
  });
});
