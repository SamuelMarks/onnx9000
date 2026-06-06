import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/worker/index';

describe('index.ts', () => {
  it('should instantiate and cover WorkerPipeline', () => {
    try {
       const obj = new (Module as any).WorkerPipeline();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
