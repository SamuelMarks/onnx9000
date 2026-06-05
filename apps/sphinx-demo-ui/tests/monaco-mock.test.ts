// @ts-nocheck
import { describe, it, expect } from 'vitest';

describe('monaco-mock', () => {
  it('should be valid', async () => {
    const mod = await import('../tests/monaco-mock.js');
    expect(mod.editor).toBeDefined();
  });
});
