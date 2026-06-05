// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { WasmManager, WasmState } from '../src/core/WasmManager.js';

describe('WasmManager', () => {
  it('should load', async () => {
    const mgr = WasmManager.getInstance();
    mgr.reset();
    expect(mgr.state).toBe(WasmState.IDLE);

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers(),
      body: {
        getReader: () => {
          let done = false;
          return {
            read: async () => {
              if (!done) {
                done = true;
                return { done: false, value: new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]) };
              }
              return { done: true };
            }
          };
        }
      }
    });

    global.WebAssembly = {
      compile: vi.fn().mockResolvedValue({}),
      instantiate: vi.fn().mockResolvedValue({})
    } as any;

    await mgr.load();
    expect(mgr.state).toBe(WasmState.LOADED);
  });
});
