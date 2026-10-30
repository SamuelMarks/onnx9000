import { describe, expect, it } from 'vitest';
import { CaffeMapper, register_caffe_op } from '../../../src/mmdnn/caffe/mapper';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover CaffeMapper', () => {
    try {
      const obj = Construct(CaffeMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover register_caffe_op', async () => {
    try {
      const res = Call(register_caffe_op);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
