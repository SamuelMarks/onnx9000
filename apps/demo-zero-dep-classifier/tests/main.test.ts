import { describe, it, expect, vi } from 'vitest';
import { initZeroDepClassifierDemo } from '../src/main.js';

describe('demo-zero-dep-classifier', () => {
  it('should run classification', () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="runBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
    `;
    initZeroDepClassifierDemo();
    document.getElementById('runBtn')?.click();
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      'Pipeline finished successfully.',
    );

    document.getElementById('resetBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Ready.');
    vi.useRealTimers();
  });
});
