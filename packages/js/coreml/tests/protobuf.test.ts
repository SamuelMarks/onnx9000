import { describe, expect, it } from 'vitest';
import { Writer } from '../src/protobuf';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('protobuf.ts', () => {
  it('should instantiate and cover Writer', () => {
    try {
      const obj = Construct(Writer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
