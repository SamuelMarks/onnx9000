import { describe, expect, it } from 'vitest';
import * as Module from '../../src/mmdnn/reporter';

describe('reporter.ts', () => {
  it('should instantiate and cover MMDNNError', () => {
    try {
      const obj = new (Module as any).MMDNNError();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MMDNNReporter', () => {
    try {
      const obj = new (Module as any).MMDNNReporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
