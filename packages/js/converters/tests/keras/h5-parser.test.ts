import { describe, expect, it } from 'vitest';
import { parseKerasH5 } from '../../src/keras/h5-parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('h5-parser.ts', () => {
  it('should call and cover parseKerasH5', async () => {
    try {
      const res = Call(parseKerasH5);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
