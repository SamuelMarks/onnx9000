// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOptimizeDemo } from '../src/main.js';

describe('demo-optimize', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><input id="passes"/><button id="optimizeBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should run graph optimization and reset', async () => {
    document.body.innerHTML = '';
    initOptimizeDemo();

    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><input id="passes"/><button id="optimizeBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initOptimizeDemo();

    const optimizeBtn = document.getElementById('optimizeBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    optimizeBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('Graph optimization complete.');

    resetBtn.click();
    expect(out.textContent).toContain('Waiting to optimize...');
  });
});
