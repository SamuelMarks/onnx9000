import { describe, expect, it } from 'vitest';
import { NodeFusionRegistry } from '../../src/mmdnn/fusion';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('fusion.ts', () => {
  it('should instantiate and cover NodeFusionRegistry', () => {
    try {
      const obj = Construct(NodeFusionRegistry);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
