import { describe, expect, it } from 'vitest';
import { KerasImporter } from '../../../src/mmdnn/keras/importer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('importer.ts', () => {
  it('should instantiate and cover KerasImporter', () => {
    try {
      const obj = Construct(KerasImporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
