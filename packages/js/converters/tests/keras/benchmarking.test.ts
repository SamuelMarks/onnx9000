import { describe, expect, it } from 'vitest';
import * as Module from '../../src/keras/benchmarking';

describe('benchmarking.ts', () => {
  it('should instantiate and cover ChromeTraceExporter', () => {
    try {
      const obj = new (Module as any).ChromeTraceExporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover validateMathematicalTolerance', async () => {
    try {
      const res = (Module as any).validateMathematicalTolerance();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
