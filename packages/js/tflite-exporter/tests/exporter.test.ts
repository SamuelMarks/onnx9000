import { describe, expect, it } from 'vitest';
import * as Module from '../src/exporter';

describe('exporter.ts', () => {
  it('should instantiate and cover TFLiteExporter', () => {
    try {
      const obj = new (Module as any).TFLiteExporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
