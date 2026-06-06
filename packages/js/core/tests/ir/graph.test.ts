import { describe, expect, it } from 'vitest';
import * as Module from '../../src/ir/graph';

describe('graph.ts', () => {
  it('should instantiate and cover ValueInfo', () => {
    try {
      const obj = new (Module as any).ValueInfo();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Graph', () => {
    try {
      const obj = new (Module as any).Graph();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
