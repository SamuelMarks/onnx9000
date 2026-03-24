/**
 * Base Component class for pure Vanilla JS DOM UI architecture.
 * Provides a highly controlled, lightweight lifecycle: mount, render, and unmount.
 */
export abstract class Component<T extends HTMLElement = HTMLElement> {
  public element!: T;
  private cleanupFunctions: Array<() => void> = [];

  /**
   * Abstract `render` returns a single root DOM element or fragment that represents this component.
   *
   * @returns The newly created root HTMLElement for this component.
   */
  protected abstract render(): T;

  /**
   * Mounts the component onto a DOM parent node.
   *
   * @param parent - The DOM node where the component element is appended.
   */
  public mount(parent: HTMLElement): void {
    if (!this.element) {
      throw new Error(
        'Component element not initialized. Did you forget to set this.element in the constructor?'
      );
    }
    parent.appendChild(this.element);
    this.onMount();
  }

  /**
   * Replaces an existing DOM node with this component's element.
   *
   * @param oldElement - The DOM node to replace.
   */
  public replace(oldElement: HTMLElement): void {
    if (!this.element) {
      throw new Error(
        'Component element not initialized. Did you forget to set this.element in the constructor?'
      );
    }
    if (oldElement.parentNode) {
      oldElement.parentNode.replaceChild(this.element, oldElement);
      this.onMount();
    }
  }

  /**
   * Optional lifecycle hook invoked immediately after the element is attached to the DOM.
   * Useful for attaching event listeners that depend on DOM layout.
   */
  protected onMount(): void {}

  /**
   * Lifecycle hook invoked to completely unmount the component and destroy bindings.
   * Will remove the element from the DOM and run any functions pushed into `cleanupFunctions`.
   */
  public unmount(): void {
    // Run cleanup tasks
    for (const cleanup of this.cleanupFunctions) {
      cleanup();
    }
    this.cleanupFunctions = [];

    // Remove element if it exists in DOM
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }

  /**
   * Helper to bind DOM events while ensuring they are properly removed on unmount.
   *
   * @param target - The DOM element or window object emitting the event.
   * @param event - The name of the event (e.g., 'click').
   * @param handler - The listener function.
   * @param options - Additional options to pass to addEventListener.
   */
  protected addDOMListener(
    target: EventTarget,
    event: string,
    handler: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions
  ): void {
    target.addEventListener(event, handler, options);
    this.cleanupFunctions.push(() => {
      target.removeEventListener(event, handler, options);
    });
  }

  /**
   * Registers a callback function to be called when `unmount()` is executed.
   * Useful for cleaning up interval timers, state listeners, or subscriptions.
   *
   * @param fn - The cleanup function.
   */
  protected onCleanup(fn: () => void): void {
    this.cleanupFunctions.push(fn);
  }
}
