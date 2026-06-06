// @ts-nocheck
import { describe, expect, it } from 'vitest';

describe('monaco-mock', () => {
  it('should be valid', async () => {
    const mod = await import('../tests/monaco-mock.js');
    expect(mod.editor).toBeDefined();
  });
});
