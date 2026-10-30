// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initMlirLoweringDemo } from '../src/main.js';

describe('demo-mlir-lowering', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="lowerBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should execute all 5 MLIR lowering stages and reset', async () => {
    document.body.innerHTML = '';
    initMlirLoweringDemo();

    document.body.innerHTML =
      '<button id="lowerBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initMlirLoweringDemo();

    const lowerBtn = document.getElementById('lowerBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    lowerBtn.click();
    await new Promise((r) => setTimeout(r, 250));
    expect(out.textContent).toContain('MLIR Lowering Pipeline Completed Successfully!');

    resetBtn.click();
    expect(out.textContent).toContain('Ready to compile.');
  });
});
