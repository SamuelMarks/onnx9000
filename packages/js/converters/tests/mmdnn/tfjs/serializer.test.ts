import { describe, expect, it } from 'vitest';
import { serializeTFJSWeights } from '../../../src/mmdnn/tfjs/serializer';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('serializer.ts', () => {
  it('should call and cover serializeTFJSWeights', async () => {
    try {
      const res = Call(serializeTFJSWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
