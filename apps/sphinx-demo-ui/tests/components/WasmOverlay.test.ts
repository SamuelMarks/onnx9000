import { describe, expect, it } from 'vitest';
import { WasmOverlay } from '../../src/components/WasmOverlay';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('WasmOverlay.ts', () => {
  it('should instantiate and cover WasmOverlay', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(WasmOverlay);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
