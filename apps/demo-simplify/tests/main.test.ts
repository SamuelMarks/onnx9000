// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initSimplifyDemo } from '../src/main.js';

describe('demo-simplify', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><button id="simplifyBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should simplify graph and reset', async () => {
    document.body.innerHTML = '';
    initSimplifyDemo();

    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><button id="simplifyBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initSimplifyDemo();

    const simplifyBtn = document.getElementById('simplifyBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    simplifyBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('Graph simplification complete.');

    resetBtn.click();
    expect(out.textContent).toContain('Waiting to simplify...');
  });
});
