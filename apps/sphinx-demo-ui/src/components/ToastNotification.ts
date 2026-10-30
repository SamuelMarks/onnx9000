/**
 * @fileoverview ToastNotification.ts
 * Provides ToastNotification functionality for the Sphinx Demo UI.
 */
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';

/** Interface `ToastMessage` definition. */
export interface ToastMessage {
  /** Property `message`. */
  message: string;
  /** Property `type`. */
  type: 'success' | 'error' | 'info';
  /** Property `durationMs`. */
  durationMs?: number;
}

/** Represents the `ToastNotification` class. */
export class ToastNotification extends Component<HTMLDivElement> {
  /** Property `timeoutId` of `ToastNotification`. */
  private timeoutId: ReturnType<typeof setTimeout> | null = null;

  /** Initializes a new `ToastNotification` instance. */
  constructor() {
    super();
    this.element = this.render();
  }

  /**
   * Executes `render` on `ToastNotification`.
   * @returns Result of `render`.
   */
  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-toast-container';
    container.style.display = 'none';
    return container;
  }

  /**
   * Executes `show` on `ToastNotification`.
   * @param toast Parameter `toast`.
   */
  public show(toast: ToastMessage): void {
    if (this.timeoutId) clearTimeout(this.timeoutId);

    this.element.textContent = toast.message;
    this.element.className = `demo-toast-container demo-toast-${toast.type}`;
    this.element.style.display = 'block';

    const duration = toast.durationMs || 3000;
    this.timeoutId = setTimeout(() => {
      this.element.style.display = 'none';
      this.timeoutId = null;
    }, duration);
  }

  /** Executes `onMount` on `ToastNotification`. */
  protected onMount(): void {
    this.onCleanup(
      globalEventBus.on<ToastMessage>('SHOW_TOAST', (toast) => {
        this.show(toast);
      }),
    );
  }
}
