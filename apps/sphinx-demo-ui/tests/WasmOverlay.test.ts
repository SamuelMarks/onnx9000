// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { WasmOverlay } from '../src/components/WasmOverlay.js';
import { globalEventBus } from '../src/core/EventBus.js';

global.fetch = vi.fn().mockResolvedValue({ headers: new Headers() });

vi.mock('../src/core/WasmManager.js', () => ({
  WasmState: { ERROR: 'error', LOADED: 'loaded' },
  WasmManager: {
    getInstance: vi.fn().mockReturnValue({ load: vi.fn() }),
  },
}));

describe('WasmOverlay', () => {
  it('should render and load', async () => {
    const overlay = new WasmOverlay();
    overlay.mount(document.body);

    const loadBtn = overlay.element.querySelector('.demo-btn-primary') as HTMLButtonElement;
    expect(loadBtn).not.toBeNull();

    loadBtn.click();
    await new Promise((r) => setTimeout(r, 10));

    globalEventBus.emit('WASM_PROGRESS', 50);
    globalEventBus.emit('WASM_STATE_CHANGED', 'loaded');

    await new Promise((r) => setTimeout(r, 400));
    expect(overlay.element.parentNode).toBeNull();
  });
});
