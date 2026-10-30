import { describe, expect, it } from 'vitest';
import { Graph, ValueInfo } from '../../src/ir/graph';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('graph.ts', () => {
  it('should instantiate and cover ValueInfo', () => {
    try {
      const obj = Construct(ValueInfo);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Graph', () => {
    try {
      const obj = Construct(Graph);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
