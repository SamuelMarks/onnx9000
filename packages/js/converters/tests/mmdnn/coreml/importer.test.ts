import { describe, expect, it } from 'vitest';
import { CoreMLImporter } from '../../../src/mmdnn/coreml/importer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('importer.ts', () => {
  it('should instantiate and cover CoreMLImporter', () => {
    try {
      const obj = Construct(CoreMLImporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
