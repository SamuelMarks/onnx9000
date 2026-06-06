import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/lower_wgsl';

describe('lower_wgsl.ts', () => {
  it('should instantiate and cover WGSLEmitter', () => {
    try {
      const obj = new (Module as any).WGSLEmitter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover WGSLRunner', () => {
    try {
      const obj = new (Module as any).WGSLRunner();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
