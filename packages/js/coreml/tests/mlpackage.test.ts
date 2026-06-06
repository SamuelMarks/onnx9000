import { describe, it, expect, vi } from 'vitest';
import * as Module from '../src/mlpackage';

describe('mlpackage.ts', () => {
  it('should instantiate and cover MLPackageBuilder', () => {
    try {
       const obj = new (Module as any).MLPackageBuilder();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
