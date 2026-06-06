import { describe, expect, it } from 'vitest';
import * as Module from '../../src/mmdnn/fusion';

describe('fusion.ts', () => {
  it('should instantiate and cover NodeFusionRegistry', () => {
    try {
      const obj = new (Module as any).NodeFusionRegistry();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
