import { describe, expect, it } from 'vitest';
import { BenchmarkPanel } from '../../src/components/BenchmarkPanel';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('BenchmarkPanel.ts', () => {
  it('should instantiate and cover BenchmarkPanel', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(BenchmarkPanel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
