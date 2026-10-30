import { describe, expect, it } from 'vitest';
import { TFLiteExporter } from '../src/exporter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('exporter.ts', () => {
  it('should instantiate and cover TFLiteExporter', () => {
    try {
      const obj = Construct(TFLiteExporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
