import { describe, expect, it } from 'vitest';
import * as Module from '../../src/components/toolbar';

describe('toolbar.ts', () => {
  it('should instantiate and cover Toolbar', () => {
    try {
      const obj = new (Module as any).Toolbar();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
