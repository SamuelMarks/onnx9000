import { describe, it, expect, vi } from 'vitest';
import { initSimplifyDemo } from '../src/main.js';

describe('demo-simplify', () => {
  it('should run simplify', () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="simplifyBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
    `;
    initSimplifyDemo();
    document.getElementById('simplifyBtn')?.click();
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      'Graph simplification complete',
    );

    document.getElementById('resetBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Waiting to simplify');
    vi.useRealTimers();
  });
});
