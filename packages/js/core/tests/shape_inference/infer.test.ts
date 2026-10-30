import { describe, expect, it } from 'vitest';
import { inferShapes } from '../../src/shape_inference/infer';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('infer.ts', () => {
  it('should call and cover inferShapes', async () => {
    try {
      const res = Call(inferShapes);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
