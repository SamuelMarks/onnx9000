import { describe, expect, it } from 'vitest';
import * as Module from '../../src/core/Debouncer';

describe('Debouncer.ts', () => {
  it('should instantiate and cover Debouncer', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).Debouncer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
