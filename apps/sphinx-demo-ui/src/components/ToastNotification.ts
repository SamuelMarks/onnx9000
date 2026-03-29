/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';

export interface ToastMessage {
  message: string;
  type: 'success' | 'error' | 'info';
  durationMs?: number;
}

export class ToastNotification extends Component<HTMLDivElement> {
  private timeoutId: object = null;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-toast-container';
    container.style.display = 'none';
    return container;
  }

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

  protected onMount(): void {
    this.onCleanup(
      globalEventBus.on<ToastMessage>('SHOW_TOAST', (toast) => {
        this.show(toast);
      })
    );
  }
}
