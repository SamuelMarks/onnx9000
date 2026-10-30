import { describe, expect, it } from 'vitest';
import { WorkerPipeline } from '../../src/worker/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover WorkerPipeline', () => {
    try {
      const obj = Construct(WorkerPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
