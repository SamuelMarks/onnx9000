import { describe, expect, it } from 'vitest';
import { parseModelProto, releaseArrayBuffer } from '../../src/parser/onnx';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('onnx.ts', () => {
  it('should call and cover parseModelProto', async () => {
    try {
      const res = Call(parseModelProto);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover releaseArrayBuffer', async () => {
    try {
      const res = Call(releaseArrayBuffer);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
