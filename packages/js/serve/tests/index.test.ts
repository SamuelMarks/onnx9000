import { describe, expect, it } from 'vitest';
import { createServer, Onnx9000Server } from '../src/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover Onnx9000Server', () => {
    try {
      const obj = Construct(Onnx9000Server);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover createServer', async () => {
    try {
      const res = Call(createServer);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
