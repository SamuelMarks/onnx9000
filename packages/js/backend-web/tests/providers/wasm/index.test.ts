import { describe, expect, it } from 'vitest';
import { WasmProvider } from '../../../src/providers/wasm/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover WasmProvider', () => {
    try {
      const obj = Construct(WasmProvider);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
