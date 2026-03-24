import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WasmManager, WasmState } from '../../src/core/WasmManager';
import { globalEventBus } from '../../src/core/EventBus';

global.WebAssembly = {
  compile: vi.fn().mockResolvedValue({}),
  instantiate: vi.fn().mockResolvedValue({ exports: {} })
} as unknown as typeof WebAssembly;

describe('WasmManager fallback error', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    WasmManager.getInstance().reset();
    globalEventBus.clearAll();
  });

  it('should handle thrown string errors (no err.message)', async () => {
    const manager = WasmManager.getInstance();
    global.fetch = vi.fn().mockRejectedValue('String Error');

    await manager.load();

    expect(manager.state).toBe(WasmState.ERROR);
    expect(manager.error).toBe('Failed to load WASM binary'); // Default fallback
  });
});

it('should cover the constructor being private', () => {
  // Reset singleton specifically
  (WasmManager as any).instance = undefined;
  const inst = WasmManager.getInstance();
  expect(inst).toBeDefined();
});
