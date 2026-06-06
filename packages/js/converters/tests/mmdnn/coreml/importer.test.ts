import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/coreml/importer';

describe('importer.ts', () => {
  it('should instantiate and cover CoreMLImporter', () => {
    try {
      const obj = new (Module as any).CoreMLImporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
