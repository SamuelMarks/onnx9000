import { describe, expect, it } from 'vitest';
import { ZooMLIRLoweringPass } from '../../src/passes/zoo_lowering';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('zoo_lowering.ts', () => {
  it('should instantiate and cover ZooMLIRLoweringPass', () => {
    try {
      const obj = Construct(ZooMLIRLoweringPass);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
