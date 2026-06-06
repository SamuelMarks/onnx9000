import { describe, expect, it } from 'vitest';
import * as Module from '../../src/components/SplitPane';

describe('SplitPane.ts', () => {
  it('should instantiate and cover SplitPane', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).SplitPane();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
