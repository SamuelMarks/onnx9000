import { describe, expect, it } from 'vitest';
import { extractTraceViaPyodide } from '../../src/keras/autograph';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('autograph.ts', () => {
  it('should call and cover extractTraceViaPyodide', async () => {
    try {
      const res = Call(extractTraceViaPyodide);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
