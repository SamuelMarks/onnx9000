import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PromoteButton } from '../../src/components/PromoteButton';
import { globalEventBus } from '../../src/core/EventBus';

describe('PromoteButton', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
  });

  it('should render disabled initially', () => {
    const btn = new PromoteButton();
    const el = (btn as any).element as HTMLButtonElement;
    expect(el.className).toBe('demo-btn-promote');
    expect(el.disabled).toBe(true);
  });

  it('should enable when artifact is generated', () => {
    const btn = new PromoteButton();
    const el = (btn as any).element as HTMLButtonElement;
    btn.mount(document.body);

    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(el.disabled).toBe(false);
  });

  it('should disable when target is cleared', () => {
    const btn = new PromoteButton();
    const el = (btn as any).element as HTMLButtonElement;
    btn.mount(document.body);

    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(el.disabled).toBe(false);

    globalEventBus.emit('TARGET_CLEARED', {});
    expect(el.disabled).toBe(true);
  });

  it('should emit PROMOTE event when clicked if enabled', () => {
    const btn = new PromoteButton();
    const el = (btn as any).element as HTMLButtonElement;
    btn.mount(document.body);

    const promoteSpy = vi.fn();
    globalEventBus.on('PROMOTE_TARGET_TO_SOURCE', promoteSpy);

    // Clicking when disabled does nothing
    el.click();
    expect(promoteSpy).not.toHaveBeenCalled();

    // Enable it
    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(el.disabled).toBe(false);

    // Click when enabled
    el.click();
    expect(promoteSpy).toHaveBeenCalledTimes(1);
  });

  it('should handle WASM processing state', () => {
    const btn = new PromoteButton();
    const el = (btn as any).element as HTMLButtonElement;
    btn.mount(document.body);

    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(el.disabled).toBe(false);

    // Start processing
    globalEventBus.emit('WASM_PROCESSING_START', {});
    expect(el.disabled).toBe(true);

    // Try to generate an artifact during processing (should remain disabled)
    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(el.disabled).toBe(true);

    // End processing (success)
    globalEventBus.emit('WASM_PROCESSING_END', true);
    expect(el.disabled).toBe(false);

    // End processing (fail)
    globalEventBus.emit('WASM_PROCESSING_START', {});
    globalEventBus.emit('WASM_PROCESSING_END', false);
    expect(el.disabled).toBe(true);
  });
});
