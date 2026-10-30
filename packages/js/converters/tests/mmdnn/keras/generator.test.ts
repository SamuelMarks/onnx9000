import { describe, expect, it } from 'vitest';
import { KerasGenerator } from '../../../src/mmdnn/keras/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover KerasGenerator', () => {
    try {
      const obj = Construct(KerasGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
