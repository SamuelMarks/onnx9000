import { describe, it, expect, vi } from 'vitest';
import { initOptimizeDemo } from '../src/main.js';

describe('demo-optimize', () => {
  it('should run optimization', () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="optimizeBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
    `;
    initOptimizeDemo();
    document.getElementById('optimizeBtn')?.click();
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('Graph optimization complete');

    document.getElementById('resetBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Waiting to optimize');
    vi.useRealTimers();
  });
});
