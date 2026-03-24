import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ToastNotification } from '../../src/components/ToastNotification';
import { globalEventBus } from '../../src/core/EventBus';

describe('ToastNotification', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    globalEventBus.clearAll();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should render hidden initially', () => {
    const toast = new ToastNotification();
    const el = (toast as any).element as HTMLElement;
    expect(el.className).toBe('demo-toast-container');
    expect(el.style.display).toBe('none');
  });

  it('should show toast message on SHOW_TOAST event', () => {
    const toast = new ToastNotification();
    const el = (toast as any).element as HTMLElement;
    toast.mount(document.body);

    globalEventBus.emit('SHOW_TOAST', { message: 'Test message', type: 'success' });

    expect(el.style.display).toBe('block');
    expect(el.textContent).toBe('Test message');
    expect(el.className).toContain('demo-toast-success');
  });

  it('should hide toast message after duration', () => {
    const toast = new ToastNotification();
    const el = (toast as any).element as HTMLElement;
    toast.mount(document.body);

    globalEventBus.emit('SHOW_TOAST', { message: 'Test', type: 'info', durationMs: 1000 });

    expect(el.style.display).toBe('block');

    vi.advanceTimersByTime(1000);
    expect(el.style.display).toBe('none');
  });

  it('should clear existing timeout if new toast appears', () => {
    const toast = new ToastNotification();
    const el = (toast as any).element as HTMLElement;
    toast.mount(document.body);

    globalEventBus.emit('SHOW_TOAST', { message: 'First', type: 'info', durationMs: 5000 });

    vi.advanceTimersByTime(2000);

    // New toast before first one fades out
    globalEventBus.emit('SHOW_TOAST', { message: 'Second', type: 'error', durationMs: 1000 });

    vi.advanceTimersByTime(2000); // Original would have triggered, new one should trigger here

    expect(el.textContent).toBe('Second');
    expect(el.style.display).toBe('none'); // Second timeout finished
  });
});
