// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { ToastNotification } from '../src/components/ToastNotification.js';

describe('ToastNotification', () => {
  it('should show and hide', () => {
    vi.useFakeTimers();
    const toast = new ToastNotification();
    document.body.appendChild(toast.element);

    toast.show({ message: 'test', type: 'info', durationMs: 1000 });
    expect(toast.element.style.display).toBe('block');
    expect(toast.element.textContent).toBe('test');

    vi.runAllTimers();
    expect(toast.element.style.display).toBe('none');
    vi.useRealTimers();
  });
});
