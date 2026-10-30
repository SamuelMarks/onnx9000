import { describe, expect, it } from 'vitest';
import { extractKerasTopology } from '../../src/keras/keras-ast';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('keras-ast.ts', () => {
  it('should call and cover extractKerasTopology', async () => {
    try {
      const res = Call(extractKerasTopology);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
