import { describe, it, expect, vi } from 'vitest';
import { TfjsShimDemoElement } from '../src/ui';

describe('Coverage UI', () => {
  it('TfjsShimDemoElement', () => {
    const el = new TfjsShimDemoElement();

    let html = '';
    let clickListener: Object;
    const mockShadow = {
      innerHTML: '',
      querySelector: (sel: string) => {
        if (sel === '#run-btn') {
          return {
            addEventListener: (evt: string, cb: Object) => {
              clickListener = cb;
            },
          };
        }
        if (sel === '#results') {
          return { textContent: '' };
        }
      },
    };

    // vitest with jsdom already provides HTMLElement and attachShadow, so let's try standard DOM API
    if (typeof document !== 'undefined') {
      document.body.appendChild(el);
      const btn = el.shadowRoot!.querySelector('#run-btn') as HTMLButtonElement;
      btn.click();
      const results = el.shadowRoot!.querySelector('#results');
      expect(results?.textContent).toContain('Results match!');
    }
  });
});
