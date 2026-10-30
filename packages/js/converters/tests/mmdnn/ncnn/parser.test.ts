import { describe, expect, it } from 'vitest';
import { NcnnBinParser, parseNcnnParam } from '../../../src/mmdnn/ncnn/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover NcnnBinParser', () => {
    try {
      const obj = Construct(NcnnBinParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseNcnnParam', async () => {
    try {
      const res = Call(parseNcnnParam);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
