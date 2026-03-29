/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WasmOverlay } from '../../src/components/WasmOverlay';
import { WasmManager, WasmState } from '../../src/core/WasmManager';
import { globalEventBus } from '../../src/core/EventBus';

global.WebAssembly = {
  compile: vi.fn().mockResolvedValue({}),
  instantiate: vi.fn().mockResolvedValue({ exports: {} })
} as object as typeof WebAssembly;

describe('WasmOverlay', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    WasmManager.getInstance().reset();
    globalEventBus.clearAll();
  });

  it('should render overlay components properly', () => {
    const overlay = new WasmOverlay();
    // Expose DOM element via any since it's an abstract class protected prop
    const el = (overlay as object).element as HTMLElement;

    expect(el.className).toBe('demo-wasm-overlay');
    expect(el.querySelector('.demo-wasm-modal')).not.toBeNull();
    expect(el.querySelector('.demo-btn-primary')).not.toBeNull();
    expect(el.querySelector('.demo-wasm-progress-container')).not.toBeNull();
    expect(el.querySelector('.demo-wasm-error-text')).not.toBeNull();
  });

  it('should trigger WasmManager.load on click and update UI', () => {
    const overlay = new WasmOverlay();
    const parent = document.createElement('div');
    overlay.mount(parent);

    const el = (overlay as object).element as HTMLElement;
    const btn = el.querySelector('.demo-btn-primary') as HTMLButtonElement;
    const progressContainer = el.querySelector('.demo-wasm-progress-container') as HTMLDivElement;

    const loadSpy = vi.spyOn(WasmManager.getInstance(), 'load').mockResolvedValue();

    expect(progressContainer.style.display).toBe('none');

    // Dispatch click
    btn.click();

    expect(loadSpy).toHaveBeenCalledTimes(1);
    expect(btn.style.display).toBe('none');
    expect(progressContainer.style.display).toBe('block');
  });

  it('should update progress bar on WASM_PROGRESS event', () => {
    const overlay = new WasmOverlay();
    const parent = document.createElement('div');
    overlay.mount(parent);

    const el = (overlay as object).element as HTMLElement;
    const progressBar = el.querySelector('.demo-wasm-progress-bar') as HTMLDivElement;
    const progressText = el.querySelector('.demo-wasm-progress-text') as HTMLParagraphElement;

    globalEventBus.emit('WASM_PROGRESS', 42);

    expect(progressBar.style.width).toBe('42%');
    expect(progressText.textContent).toBe('42%');
  });

  it('should show error message on WASM_STATE_CHANGED ERROR', () => {
    const overlay = new WasmOverlay();
    const parent = document.createElement('div');
    overlay.mount(parent);

    const el = (overlay as object).element as HTMLElement;
    const errorText = el.querySelector('.demo-wasm-error-text') as HTMLParagraphElement;
    const btn = el.querySelector('.demo-btn-primary') as HTMLButtonElement;

    // Simulate error state
    vi.spyOn(WasmManager.getInstance(), 'error', 'get').mockReturnValue('Simulated fetch failure');
    globalEventBus.emit('WASM_STATE_CHANGED', WasmState.ERROR);

    expect(errorText.style.display).toBe('block');
    expect(errorText.textContent).toBe('Simulated fetch failure');
    expect(btn.style.display).toBe('block'); // Button returns so user can retry
  });

  it('should fade out and unmount on WASM_STATE_CHANGED LOADED', () => {
    vi.useFakeTimers();
    const overlay = new WasmOverlay();
    const parent = document.createElement('div');
    overlay.mount(parent);

    const el = (overlay as object).element as HTMLElement;

    globalEventBus.emit('WASM_STATE_CHANGED', WasmState.LOADED);

    expect(el.style.opacity).toBe('0');

    // Fast-forward the 300ms setTimeout
    vi.runAllTimers();

    // Element should be removed from parent
    expect(parent.children.length).toBe(0);

    vi.useRealTimers();
  });
});

it('should fallback to Unknown error on WASM_STATE_CHANGED ERROR if no manager error', () => {
  const overlay = new WasmOverlay();
  const parent = document.createElement('div');
  overlay.mount(parent);

  const el = (overlay as object).element as HTMLElement;
  const errorText = el.querySelector('.demo-wasm-error-text') as HTMLParagraphElement;

  // Simulate error state without setting error text on the manager
  vi.spyOn(WasmManager.getInstance(), 'error', 'get').mockReturnValue(null);
  globalEventBus.emit('WASM_STATE_CHANGED', WasmState.ERROR);

  expect(errorText.textContent).toBe('Unknown error occurred.');
});
