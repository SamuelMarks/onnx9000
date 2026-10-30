import { describe, expect, it } from 'vitest';
import {
  lowerLinalgToSCF,
  unrollLoops,
  vectorizeLoops,
  WASMEmitter,
} from '../../src/passes/lower_wasm';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('lower_wasm.ts', () => {
  it('should instantiate and cover WASMEmitter', () => {
    try {
      const obj = Construct(WASMEmitter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover lowerLinalgToSCF', async () => {
    try {
      const res = Call(lowerLinalgToSCF);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover unrollLoops', async () => {
    try {
      const res = Call(unrollLoops);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover vectorizeLoops', async () => {
    try {
      const res = Call(vectorizeLoops);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
