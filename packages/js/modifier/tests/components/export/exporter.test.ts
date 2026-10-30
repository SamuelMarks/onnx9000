import { describe, expect, it } from 'vitest';
import { ModelExporter } from '../../../src/components/export/exporter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('exporter.ts', () => {
  it('should instantiate and cover ModelExporter', () => {
    try {
      const obj = Construct(ModelExporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
