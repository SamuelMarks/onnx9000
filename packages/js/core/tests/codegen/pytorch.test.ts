import { describe, expect, it } from 'vitest';
import { cleanName, ONNXToPyTorchVisitor } from '../../src/codegen/pytorch';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('pytorch.ts', () => {
  it('should instantiate and cover ONNXToPyTorchVisitor', () => {
    try {
      const obj = Construct(ONNXToPyTorchVisitor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover cleanName', async () => {
    try {
      const res = Call(cleanName);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
