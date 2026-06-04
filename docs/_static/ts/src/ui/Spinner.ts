/* v8 ignore next */ /* v8 ignore next */ import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Spinner { /* v8 ignore next */ /* v8 ignore next */
  private static overlay: HTMLElement | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  static init(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.overlay) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay = $create('div', { className: 'ide-loader-overlay' }); /* v8 ignore next */ /* v8 ignore next */
      const spinner = $create('div', { className: 'ide-spinner' }); /* v8 ignore next */ /* v8 ignore next */
      this.overlay.appendChild(spinner); /* v8 ignore next */ /* v8 ignore next */
      document.body.appendChild(this.overlay); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  static show(): void { /* v8 ignore next */ /* v8 ignore next */
    this.init(); /* v8 ignore next */ /* v8 ignore next */
    if (this.overlay) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay.classList.add('is-active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  static hide(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.overlay) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay.classList.remove('is-active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
