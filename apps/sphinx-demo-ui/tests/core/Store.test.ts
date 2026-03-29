/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { Store } from '../../src/core/Store';

interface TestState {
  counter: number;
  text: string;
}

describe('Store', () => {
  it('should initialize with given state', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    expect(store.state.counter).toBe(0);
    expect(store.state.text).toBe('hello');
  });

  it('should emit a specific property change event when a property is updated', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    const mockCallback = vi.fn();

    store.onPropertyChange('counter', mockCallback);

    // Act
    store.state.counter = 1;

    // Assert
    expect(mockCallback).toHaveBeenCalledTimes(1);
    expect(mockCallback).toHaveBeenCalledWith(1);
  });

  it('should emit a general change event when any property is updated', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    const mockCallback = vi.fn();

    store.onChange(mockCallback);

    // Act
    store.state.text = 'world';

    // Assert
    expect(mockCallback).toHaveBeenCalledTimes(1);
    expect(mockCallback).toHaveBeenCalledWith({ property: 'text', value: 'world' });
  });

  it('should not emit an event if the property value does not change', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    const mockCallback = vi.fn();

    store.onPropertyChange('counter', mockCallback);

    // Act
    store.state.counter = 0; // Same value

    // Assert
    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('should allow unsubscribing from property changes', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    const mockCallback = vi.fn();

    const unsubscribe = store.onPropertyChange('counter', mockCallback);
    unsubscribe();

    store.state.counter = 1;

    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('should allow unsubscribing from general changes', () => {
    const store = new Store<TestState>({ counter: 0, text: 'hello' });
    const mockCallback = vi.fn();

    const unsubscribe = store.onChange(mockCallback);
    unsubscribe();

    store.state.text = 'world';

    expect(mockCallback).not.toHaveBeenCalled();
  });
});
