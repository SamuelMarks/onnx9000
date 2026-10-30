// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initMobileMemoryDemo } from '../src/main.js';

describe('demo-mobile-memory', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="allocateBtn"></button><button id="runInferenceBtn"></button><button id="freeBtn"></button><div id="arena-container"></div><div id="output"></div>';
  });

  it('should allocate, run inference in-use, and free arena blocks', async () => {
    document.body.innerHTML = '';
    initMobileMemoryDemo();

    document.body.innerHTML =
      '<button id="allocateBtn"></button><button id="runInferenceBtn"></button><button id="freeBtn"></button><div id="arena-container"></div><div id="output"></div>';
    initMobileMemoryDemo();

    const allocateBtn = document.getElementById('allocateBtn') as HTMLButtonElement;
    const runInferenceBtn = document.getElementById('runInferenceBtn') as HTMLButtonElement;
    const freeBtn = document.getElementById('freeBtn') as HTMLButtonElement;
    const arenaContainer = document.getElementById('arena-container') as HTMLDivElement;
    const out = document.getElementById('output') as HTMLDivElement;

    runInferenceBtn.click();

    allocateBtn.click();
    expect(arenaContainer.children).toHaveLength(20);
    expect(out.textContent).toContain('Arena pre-allocated successfully.');

    runInferenceBtn.click();
    expect(arenaContainer.querySelectorAll('.allocated').length).toBeGreaterThan(0);

    await new Promise((r) => setTimeout(r, 80));
    expect(out.textContent).toContain('Inference complete.');

    freeBtn.click();
    expect(arenaContainer.children).toHaveLength(0);
    expect(out.textContent).toContain('Arena memory freed.');
  });
});
