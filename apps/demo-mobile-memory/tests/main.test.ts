import { describe, it, expect, vi } from 'vitest';
import { initMobileMemoryDemo } from '../src/main.js';

describe('demo-mobile-memory', () => {
  it('should allocate and run', () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="allocateBtn"></button>
      <button id="runInferenceBtn"></button>
      <button id="freeBtn"></button>
      <div id="arena-container"></div>
      <div id="output"></div>
    `;
    initMobileMemoryDemo();

    document.getElementById('allocateBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Arena pre-allocated');

    document.getElementById('runInferenceBtn')?.click();
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('Inference complete');

    document.getElementById('freeBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Arena memory freed');

    vi.useRealTimers();
  });
});
