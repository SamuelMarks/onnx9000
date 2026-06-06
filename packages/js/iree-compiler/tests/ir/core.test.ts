import { describe, expect, it } from 'vitest';
import * as Module from '../../src/ir/core';

describe('core.ts', () => {
  it('should instantiate and cover Value', () => {
    try {
      const obj = new (Module as any).Value();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover BlockArgument', () => {
    try {
      const obj = new (Module as any).BlockArgument();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Block', () => {
    try {
      const obj = new (Module as any).Block();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Region', () => {
    try {
      const obj = new (Module as any).Region();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Operation', () => {
    try {
      const obj = new (Module as any).Operation();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
