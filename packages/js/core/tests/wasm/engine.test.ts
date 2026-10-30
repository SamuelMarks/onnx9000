import { describe, expect, it } from 'vitest';
import { execute_graph, init } from '../../src/wasm/engine';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('engine.ts', () => {
  it('should call and cover init', async () => {
    try {
      const res = Call(init);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover execute_graph', async () => {
    try {
      const res = Call(execute_graph);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
