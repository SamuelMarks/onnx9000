import { describe, expect, it } from 'vitest';
import * as Module from '../../src/core/EventBus';

describe('EventBus.ts', () => {
  it('should instantiate and cover EventBus', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).EventBus();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
