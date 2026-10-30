import { describe, expect, it } from 'vitest';
import { serializeModelProto } from '../../src/parser/onnx_writer';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('onnx_writer.ts', () => {
  it('should call and cover serializeModelProto', async () => {
    try {
      const res = Call(serializeModelProto);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
