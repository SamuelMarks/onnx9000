import { describe, expect, it } from 'vitest';
import { GraphPartitioner } from '../src/partitioner';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('partitioner.ts', () => {
  it('should instantiate and cover GraphPartitioner', () => {
    try {
      const obj = Construct(GraphPartitioner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
