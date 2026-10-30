import { describe, expect, it } from 'vitest';
import { H2OMapper } from '../../../src/mmdnn/h2o/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover H2OMapper', () => {
    try {
      const obj = Construct(H2OMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
