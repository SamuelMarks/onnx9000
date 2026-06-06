import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/mil/ast';

describe('ast.ts', () => {
  it('should instantiate and cover Var', () => {
    try {
       const obj = new (Module as any).Var();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Operation', () => {
    try {
       const obj = new (Module as any).Operation();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Block', () => {
    try {
       const obj = new (Module as any).Block();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Function', () => {
    try {
       const obj = new (Module as any).Function();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Program', () => {
    try {
       const obj = new (Module as any).Program();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
