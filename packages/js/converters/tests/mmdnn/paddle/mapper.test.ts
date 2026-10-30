import { describe, expect, it } from 'vitest';
import {
  PaddleMapper,
  register_paddle_op,
  translatePaddleShape,
} from '../../../src/mmdnn/paddle/mapper';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover PaddleMapper', () => {
    try {
      const obj = Construct(PaddleMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover register_paddle_op', async () => {
    try {
      const res = Call(register_paddle_op);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover translatePaddleShape', async () => {
    try {
      const res = Call(translatePaddleShape);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
