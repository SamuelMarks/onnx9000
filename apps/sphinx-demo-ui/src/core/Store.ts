/* eslint-disable */
// @ts-nocheck
import { EventBus } from './EventBus';

/**
 * A reactive state store using Vanilla JS Proxy.
 * Notifies an internal EventBus when properties are updated.
 *
 * Provides a highly decoupled, simple, framework-free reactivity model.
 */
export class Store<T extends object> {
  private _state: T;
  private _bus: EventBus;

  /**
   * Constructs the reactive store with initial data.
   *
   * @param initialState - The starting structure of the store.
   */
  constructor(initialState: T) {
    this._bus = new EventBus();

    // Use a proxy to intercept assignment
    this._state = new Proxy<T>(initialState, {
      set: (target: T, property: string | symbol, value: object) => {
        // Cast property to keyof T safely here since it's an object property
        const propStr = String(property);

        // Only emit if value actually changed
        if (target[propStr as keyof T] !== value) {
          target[propStr as keyof T] = value;
          // Emit change event keyed on property name
          this._bus.emit(`change:${propStr}`, value);
          // Emit a general change event
          this._bus.emit('change', { property: propStr, value });
        }
        return true;
      }
    });
  }

  /**
   * Exposes the proxied reactive state object.
   * Setting a property on this object triggers events.
   */
  public get state(): T {
    return this._state;
  }

  /**
   * Subscribe to changes on a specific property within the store.
   *
   * @param property - The property name on the state to listen to.
   * @param callback - Function invoked when the property changes.
   * @returns Unsubscribe function
   */
  public onPropertyChange<K extends keyof T>(
    property: K,
    callback: (value: T[K]) => void
  ): () => void {
    return this._bus.on(`change:${String(property)}`, callback as (payload: object) => void);
  }

  /**
   * Subscribe to any change in the store.
   *
   * @param callback - Function invoked when any property changes.
   * @returns Unsubscribe function
   */
  public onChange(callback: (event: { property: string; value: object }) => void): () => void {
    return this._bus.on('change', callback);
  }
}
