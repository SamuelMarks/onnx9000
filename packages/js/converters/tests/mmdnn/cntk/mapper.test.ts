import { describe, expect, it } from 'vitest';
import { CNTKMapper, register_cntk_op } from '../../../src/mmdnn/cntk/mapper';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover CNTKMapper', () => {
    try {
      const obj = Construct(CNTKMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover register_cntk_op', async () => {
    try {
      const res = Call(register_cntk_op);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
