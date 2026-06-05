import { describe, it, expect, vi } from 'vitest';
import { initNewModelArchDemo } from '../src/main.js';

describe('demo-new-model-arch', () => {
  it('should parse architecture', async () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="parseBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
      <textarea id="archInput"></textarea>
    `;
    initNewModelArchDemo();
    document.getElementById('parseBtn')?.click();

    for (let i = 0; i < 10; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('output')?.textContent).toContain(
      'Architecture mapped to core IR successfully',
    );
    vi.useRealTimers();
  });
});
