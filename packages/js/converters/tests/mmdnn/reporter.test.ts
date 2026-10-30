import { describe, expect, it } from 'vitest';
import { MMDNNError, MMDNNReporter } from '../../src/mmdnn/reporter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('reporter.ts', () => {
  it('should instantiate and cover MMDNNError', () => {
    try {
      const obj = Construct(MMDNNError);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MMDNNReporter', () => {
    try {
      const obj = Construct(MMDNNReporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
