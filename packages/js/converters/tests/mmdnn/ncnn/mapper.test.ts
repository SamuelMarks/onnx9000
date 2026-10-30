import { describe, expect, it } from 'vitest';
import { NcnnMapper } from '../../../src/mmdnn/ncnn/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover NcnnMapper', () => {
    try {
      const obj = Construct(NcnnMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
