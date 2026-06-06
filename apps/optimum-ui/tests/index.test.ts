import { describe, it, expect } from 'vitest';

describe('optimum-ui', () => {
  it('should export nothing', async () => {
    const exports = await import('../src/index');
    expect(Object.keys(exports)).toEqual([]);
  });
});
