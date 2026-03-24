/**
 * Simple Publish-Subscribe Event Bus for decoupled cross-component communication.
 * No external dependencies, purely vanilla JS.
 */
export class EventBus {
  /**
   * Internal storage for event subscribers.
   * Maps event names to an array of callback functions.
   */
  private listeners: Record<string, Function[]> = {};

  /**
   * Subscribes to an event with a specific callback.
   *
   * @param event - The name of the event to listen for.
   * @param callback - The function to call when the event is emitted.
   * @returns A function that when called will unsubscribe this specific callback.
   */
  public on<T = any>(event: string, callback: (payload: T) => void): () => void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);

    // Return unsubscribe function
    return () => {
      this.off(event, callback);
    };
  }

  /**
   * Unsubscribes a specific callback from an event.
   *
   * @param event - The name of the event to stop listening to.
   * @param callback - The function to remove.
   */
  public off<T = any>(event: string, callback: (payload: T) => void): void {
    if (!this.listeners[event]) {
      return;
    }
    this.listeners[event] = this.listeners[event].filter((cb) => cb !== callback);
  }

  /**
   * Emits an event, calling all registered callbacks synchronously with the provided payload.
   *
   * @param event - The name of the event to emit.
   * @param payload - The data to pass to the callbacks.
   */
  public emit<T = any>(event: string, payload?: T): void {
    if (!this.listeners[event]) {
      return;
    }
    // Clone array to avoid issues if listeners are added/removed during emit
    const listenersCpy = [...this.listeners[event]];
    for (const callback of listenersCpy) {
      callback(payload);
    }
  }

  /**
   * Clears all registered callbacks for all events. Useful for testing or cleanup.
   */
  public clearAll(): void {
    this.listeners = {};
  }
}

/**
 * Global singleton instance of EventBus.
 */
export const globalEventBus = new EventBus();
