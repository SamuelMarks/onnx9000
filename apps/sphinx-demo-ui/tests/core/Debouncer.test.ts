/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Debouncer } from '../../src/core/Debouncer';

describe('Debouncer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should debounce multiple function calls', () => {
    const debouncer = new Debouncer();
    const mockFunc = vi.fn();

    const debouncedFunc = debouncer.debounce(mockFunc, 100);

    debouncedFunc();
    debouncedFunc();
    debouncedFunc();

    expect(mockFunc).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50);
    expect(mockFunc).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50);
    expect(mockFunc).toHaveBeenCalledTimes(1);
  });

  it('should pass arguments correctly', () => {
    const debouncer = new Debouncer();
    const mockFunc = vi.fn();
    const debouncedFunc = debouncer.debounce(mockFunc, 100);

    debouncedFunc('test', 123);
    vi.advanceTimersByTime(100);

    expect(mockFunc).toHaveBeenCalledWith('test', 123);
  });

  it('should clear pending calls', () => {
    const debouncer = new Debouncer();
    const mockFunc = vi.fn();
    const debouncedFunc = debouncer.debounce(mockFunc, 100);

    debouncedFunc();
    debouncer.clear();

    vi.advanceTimersByTime(100);
    expect(mockFunc).not.toHaveBeenCalled();
  });

  it('should be safe to clear when no calls are pending', () => {
    const debouncer = new Debouncer();
    expect(() => debouncer.clear()).not.toThrow();
  });
});
