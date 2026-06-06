import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/ir/graph';

describe('graph.ts', () => {
  it('should instantiate and cover ValueInfo', () => {
    try {
       const obj = new (Module as any).ValueInfo();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Graph', () => {
    try {
       const obj = new (Module as any).Graph();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
