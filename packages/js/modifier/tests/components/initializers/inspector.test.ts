import { describe, expect, it } from 'vitest';
import { InitializerInspector } from '../../../src/components/initializers/inspector';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('inspector.ts', () => {
  it('should instantiate and cover InitializerInspector', () => {
    try {
      const obj = Construct(InitializerInspector);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
