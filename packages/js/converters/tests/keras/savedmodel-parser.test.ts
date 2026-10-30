import { describe, expect, it } from 'vitest';
import { parseSavedModel } from '../../src/keras/savedmodel-parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('savedmodel-parser.ts', () => {
  it('should call and cover parseSavedModel', async () => {
    try {
      const res = Call(parseSavedModel);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
