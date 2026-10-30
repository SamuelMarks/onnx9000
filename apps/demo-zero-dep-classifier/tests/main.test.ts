// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initZeroDepClassifierDemo } from '../src/main.js';

describe('demo-zero-dep-classifier', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="runBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should run classification pipeline and reset state', async () => {
    document.body.innerHTML = '';
    initZeroDepClassifierDemo();

    document.body.innerHTML =
      '<button id="runBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initZeroDepClassifierDemo();

    const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    runBtn.click();
    await new Promise((r) => setTimeout(r, 200));

    expect(out.textContent).toContain('Label: TABBY_CAT');
    expect(resetBtn.disabled).toBe(false);

    resetBtn.click();
    expect(out.textContent).toContain('Ready. Click "Run Classification" to start.');
    expect(runBtn.disabled).toBe(false);
  });
});
