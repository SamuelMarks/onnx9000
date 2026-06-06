import { describe, expect, it } from 'vitest';
import { TritonCompilerElement } from '../src/ui.js';

describe('TritonCompilerElement', () => {
  it('should render', () => {
    const el = new TritonCompilerElement();
    document.body.appendChild(el);
    expect(el.shadowRoot).toBeDefined();

    const genBtn = el.shadowRoot?.querySelector('#gen') as HTMLButtonElement;
    let fired = false;
    el.addEventListener('generate-requested', () => {
      fired = true;
    });
    genBtn.click();
    expect(fired).toBe(true);

    el.setCode('python', 'wgsl');
    expect(el.shadowRoot?.querySelector('#output')?.textContent).toBe('python');
  });
});
