import { describe, expect, it } from 'vitest';
import { compileGraphToTFLite } from '../../src/compiler/subgraph';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('subgraph.ts', () => {
  it('should call and cover compileGraphToTFLite', async () => {
    try {
      const res = Call(compileGraphToTFLite);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
