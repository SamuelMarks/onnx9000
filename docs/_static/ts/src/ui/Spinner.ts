import { $create } from '../core/DOM';

export class Spinner {
  private static overlay: HTMLElement | null = null;

  static init(): void {
    if (!Spinner.overlay) {
      Spinner.overlay = $create('div', { className: 'ide-loader-overlay' });
      const spinner = $create('div', { className: 'ide-spinner' });
      Spinner.overlay.appendChild(spinner);
      document.body.appendChild(Spinner.overlay);
    }
  }

  static show(): void {
    Spinner.init();
    if (Spinner.overlay) {
      Spinner.overlay.classList.add('is-active');
    }
  }

  static hide(): void {
    if (Spinner.overlay) {
      Spinner.overlay.classList.remove('is-active');
    }
  }
}
