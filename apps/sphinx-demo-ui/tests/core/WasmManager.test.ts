import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WasmManager, WasmState } from '../../src/core/WasmManager';
import { globalEventBus } from '../../src/core/EventBus';

// Mock WebAssembly so it doesn't crash in JS DOM
global.WebAssembly = {
  compile: vi.fn().mockResolvedValue({}),
  instantiate: vi.fn().mockResolvedValue({ exports: {} })
} as unknown as typeof WebAssembly;

describe('WasmManager', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    WasmManager.getInstance().reset();
    globalEventBus.clearAll();
  });

  it('should be a singleton', () => {
    const instance1 = WasmManager.getInstance();
    const instance2 = WasmManager.getInstance();
    expect(instance1).toBe(instance2);
  });

  it('should initialize with correct default state', () => {
    const manager = WasmManager.getInstance();
    expect(manager.state).toBe(WasmState.IDLE);
    expect(manager.progress).toBe(0);
    expect(manager.error).toBeNull();
  });

  it('should handle successful load with Content-Length', async () => {
    const manager = WasmManager.getInstance();

    // Mock fetch with a readable stream
    const chunks = [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])];
    let chunkIndex = 0;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-length': '5' }),
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (chunkIndex < chunks.length) {
              return { done: false, value: chunks[chunkIndex++] };
            }
            return { done: true, value: undefined };
          })
        })
      }
    } as unknown as Response);

    const stateChangedSpy = vi.fn();
    const progressSpy = vi.fn();
    const loadedSpy = vi.fn();

    globalEventBus.on('WASM_STATE_CHANGED', stateChangedSpy);
    globalEventBus.on('WASM_PROGRESS', progressSpy);
    globalEventBus.on('WASM_LOADED', loadedSpy);

    await manager.load('/fake.wasm');

    expect(global.fetch).toHaveBeenCalledWith('/fake.wasm');
    expect(manager.state).toBe(WasmState.LOADED);
    expect(manager.progress).toBe(100);
    expect(manager.error).toBeNull();

    expect(stateChangedSpy).toHaveBeenCalledWith(WasmState.LOADING);
    expect(stateChangedSpy).toHaveBeenCalledWith(WasmState.LOADED);

    expect(progressSpy).toHaveBeenCalledWith(60); // After first chunk (3/5)
    expect(progressSpy).toHaveBeenCalledWith(100); // After second chunk (5/5)

    expect(loadedSpy).toHaveBeenCalledTimes(1);
  });

  it('should fallback to fake progress without Content-Length', async () => {
    const manager = WasmManager.getInstance();

    const chunks = [new Uint8Array([1, 2, 3])];
    let chunkIndex = 0;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers(), // No content-length
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (chunkIndex < chunks.length) {
              return { done: false, value: chunks[chunkIndex++] };
            }
            return { done: true, value: undefined };
          })
        })
      }
    } as unknown as Response);

    await manager.load();
    expect(manager.progress).toBe(100);
    expect(manager.state).toBe(WasmState.LOADED);
  });

  it('should handle fetch HTTP error', async () => {
    const manager = WasmManager.getInstance();
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404 } as unknown as Response);

    await manager.load();

    expect(manager.state).toBe(WasmState.ERROR);
    expect(manager.error).toBe('HTTP error! status: 404');
  });

  it('should handle fetch network error', async () => {
    const manager = WasmManager.getInstance();
    global.fetch = vi.fn().mockRejectedValue(new Error('Network failure'));

    await manager.load();

    expect(manager.state).toBe(WasmState.ERROR);
    expect(manager.error).toBe('Network failure');
  });

  it('should ignore load if already loading', async () => {
    const manager = WasmManager.getInstance();
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {})); // Never resolves

    manager.load(); // Kick off first
    expect(manager.state).toBe(WasmState.LOADING);

    const secondFetchSpy = vi.spyOn(global, 'fetch');
    await manager.load(); // Kick off second

    expect(secondFetchSpy).not.toHaveBeenCalled();
  });

  it('should handle lack of body in fetch response', async () => {
    const manager = WasmManager.getInstance();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers(),
      body: null // Browser doesn't support streams
    } as unknown as Response);

    await manager.load();

    expect(manager.state).toBe(WasmState.ERROR);
    expect(manager.error).toBe('ReadableStream not supported by browser.');
  });
});
