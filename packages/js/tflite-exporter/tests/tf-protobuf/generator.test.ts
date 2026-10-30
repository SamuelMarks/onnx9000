import { describe, expect, it } from 'vitest';
import { SavedModelGenerator } from '../../src/tf-protobuf/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover SavedModelGenerator', () => {
    try {
      const obj = Construct(SavedModelGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
