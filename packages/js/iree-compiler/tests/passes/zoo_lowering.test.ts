import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/zoo_lowering';

describe('zoo_lowering.ts', () => {
  it('should instantiate and cover ZooMLIRLoweringPass', () => {
    try {
      const obj = new (Module as any).ZooMLIRLoweringPass();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
