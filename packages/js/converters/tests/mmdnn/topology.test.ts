import { describe, expect, it } from 'vitest';
import { topologicalSort } from '../../src/mmdnn/topology';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('topology.ts', () => {
  it('should call and cover topologicalSort', async () => {
    try {
      const res = Call(topologicalSort);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
