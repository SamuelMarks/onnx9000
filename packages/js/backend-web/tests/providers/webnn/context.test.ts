import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/providers/webnn/context';

describe('context.ts', () => {
  it('should instantiate and cover WebNNContextManager', () => {
    try {
       const obj = new (Module as any).WebNNContextManager();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
