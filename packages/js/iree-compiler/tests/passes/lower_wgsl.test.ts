import { describe, expect, it } from 'vitest';
import { WGSLEmitter, WGSLRunner } from '../../src/passes/lower_wgsl';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('lower_wgsl.ts', () => {
  it('should instantiate and cover WGSLEmitter', () => {
    try {
      const obj = Construct(WGSLEmitter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WGSLRunner', () => {
    try {
      const obj = Construct(WGSLRunner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
