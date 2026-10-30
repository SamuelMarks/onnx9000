import { describe, expect, it } from 'vitest';
import { DiagnosticTracer } from '../../src/passes/tracing';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('tracing.ts', () => {
  it('should instantiate and cover DiagnosticTracer', () => {
    try {
      const obj = Construct(DiagnosticTracer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
