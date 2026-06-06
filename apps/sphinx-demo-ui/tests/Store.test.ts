// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { Store } from '../src/core/Store.js';

describe('Store', () => {
  it('should react to changes', () => {
    const store = new Store({ count: 0 });
    const cb = vi.fn();

    store.onPropertyChange('count', cb);

    store.state.count = 1;
    expect(cb).toHaveBeenCalledWith(1);

    store.state.count = 1; // no change
    expect(cb).toHaveBeenCalledTimes(1);
  });
});
