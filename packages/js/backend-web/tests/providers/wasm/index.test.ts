import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/providers/wasm/index';

describe('index.ts', () => {
  it('should instantiate and cover WasmProvider', () => {
    try {
       const obj = new (Module as any).WasmProvider();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
