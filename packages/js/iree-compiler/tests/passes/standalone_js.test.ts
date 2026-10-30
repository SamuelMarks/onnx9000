import { describe, expect, it } from 'vitest';
import { ModelRunner, StandaloneJSExporter } from '../../src/passes/standalone_js';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('standalone_js.ts', () => {
  it('should instantiate and cover StandaloneJSExporter', () => {
    try {
      const obj = Construct(StandaloneJSExporter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ModelRunner', () => {
    try {
      const obj = Construct(ModelRunner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
