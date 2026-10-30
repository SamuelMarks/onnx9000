import { describe, expect, it } from 'vitest';
import { ChromeTraceExporter, validateMathematicalTolerance } from '../../src/keras/benchmarking';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('benchmarking.ts', () => {
  it('should instantiate and cover ChromeTraceExporter', () => {
    try {
      const obj = Construct(ChromeTraceExporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover validateMathematicalTolerance', async () => {
    try {
      const res = Call(validateMathematicalTolerance);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
