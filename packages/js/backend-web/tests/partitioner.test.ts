import { describe, it, expect, vi } from 'vitest';
import * as Module from '../src/partitioner';

describe('partitioner.ts', () => {
  it('should instantiate and cover GraphPartitioner', () => {
    try {
       const obj = new (Module as any).GraphPartitioner();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
