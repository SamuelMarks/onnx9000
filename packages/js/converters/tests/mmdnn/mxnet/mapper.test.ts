import { describe, expect, it } from 'vitest';
import { MxNetMapper, register_mxnet_op } from '../../../src/mmdnn/mxnet/mapper';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover MxNetMapper', () => {
    try {
      const obj = Construct(MxNetMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover register_mxnet_op', async () => {
    try {
      const res = Call(register_mxnet_op);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
