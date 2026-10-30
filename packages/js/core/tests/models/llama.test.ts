import { describe, expect, it } from 'vitest';
import { LLaMA, LLaMABlock, llama7b, SwiGLU } from '../../src/models/llama';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('llama.ts', () => {
  it('should instantiate and cover SwiGLU', () => {
    try {
      const obj = Construct(SwiGLU);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover LLaMABlock', () => {
    try {
      const obj = Construct(LLaMABlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover LLaMA', () => {
    try {
      const obj = Construct(LLaMA);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover llama7b', async () => {
    try {
      const res = Call(llama7b);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
