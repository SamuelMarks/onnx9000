// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { Debouncer } from '../src/core/Debouncer.js';

describe('Debouncer', () => {
  it('should debounce calls', () => {
    vi.useFakeTimers();
    const debouncer = new Debouncer();
    const fn = vi.fn();
    const wrapped = debouncer.debounce(fn, 100);

    wrapped();
    wrapped();
    wrapped();

    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(150);
    expect(fn).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });
});
