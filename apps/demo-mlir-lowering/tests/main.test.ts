import { describe, it, expect, vi } from 'vitest';
import { initMlirLoweringDemo } from '../src/main.js';

describe('demo-mlir-lowering', () => {
  it('should step through lowering', async () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="lowerBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
    `;
    initMlirLoweringDemo();
    document.getElementById('lowerBtn')?.click();

    for (let i = 0; i < 10; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('output')?.textContent).toContain(
      'MLIR Lowering Pipeline Completed Successfully!',
    );

    document.getElementById('resetBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Ready to compile');

    vi.useRealTimers();
  });
});
