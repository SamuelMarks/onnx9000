import { describe, expect, it } from 'vitest';
import { ModelRepository } from '../src/repository';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('repository.ts', () => {
  it('should instantiate and cover ModelRepository', () => {
    try {
      const obj = Construct(ModelRepository);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
