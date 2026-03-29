/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { EventBus } from '../../src/core/EventBus';

describe('EventBus', () => {
  it('should allow subscribing and emitting events', () => {
    const bus = new EventBus();
    const mockCallback = vi.fn();

    bus.on('test-event', mockCallback);
    bus.emit('test-event', { data: 123 });

    expect(mockCallback).toHaveBeenCalledTimes(1);
    expect(mockCallback).toHaveBeenCalledWith({ data: 123 });
  });

  it('should allow unsubscribing from events', () => {
    const bus = new EventBus();
    const mockCallback = vi.fn();

    const unsubscribe = bus.on('test-event', mockCallback);
    unsubscribe();

    bus.emit('test-event', { data: 123 });
    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('should handle off() directly', () => {
    const bus = new EventBus();
    const mockCallback = vi.fn();

    bus.on('test-event', mockCallback);
    bus.off('test-event', mockCallback);

    bus.emit('test-event', { data: 123 });
    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('should not throw when emitting an event with no listeners', () => {
    const bus = new EventBus();
    expect(() => {
      bus.emit('non-existent');
    }).not.toThrow();
  });

  it('should not throw when unsubscribing from an event with no listeners', () => {
    const bus = new EventBus();
    const mockCallback = vi.fn();
    expect(() => {
      bus.off('non-existent', mockCallback);
    }).not.toThrow();
  });

  it('should clear all listeners', () => {
    const bus = new EventBus();
    const mockCallback1 = vi.fn();
    const mockCallback2 = vi.fn();

    bus.on('event1', mockCallback1);
    bus.on('event2', mockCallback2);

    bus.clearAll();

    bus.emit('event1');
    bus.emit('event2');

    expect(mockCallback1).not.toHaveBeenCalled();
    expect(mockCallback2).not.toHaveBeenCalled();
  });
});
