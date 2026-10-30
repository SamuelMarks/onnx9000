import { describe, expect, it } from 'vitest';
import { WasmManager } from '../../src/core/WasmManager';

describe('WasmManager.ts', () => {
  it('should instantiate and cover WasmManager', () => {
    const obj = WasmManager.getInstance();
    expect(obj).toBeDefined();
  });
});
