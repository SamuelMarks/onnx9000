// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { EventBus } from '../src/core/EventBus.js';

describe('EventBus', () => {
  it('should sub and emit', () => {
    const bus = new EventBus();
    const cb = vi.fn();

    const unsub = bus.on('test', cb);
    bus.emit('test', 'payload');
    expect(cb).toHaveBeenCalledWith('payload');

    unsub();
    bus.emit('test', 'payload2');
    expect(cb).toHaveBeenCalledTimes(1);
  });
});
