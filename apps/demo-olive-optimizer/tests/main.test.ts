// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOliveOptimizerDemo } from '../src/main.js';

describe('demo-olive-optimizer', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
  });

  it('should initialize and run Olive optimizer demo', async () => {
    document.body.innerHTML = '';
    initOliveOptimizerDemo();

    document.body.innerHTML = '<button id="btn-run"></button>';
    initOliveOptimizerDemo();
    document.getElementById('btn-run')?.click();

    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOliveOptimizerDemo();

    const btn = document.getElementById('btn-run') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Running...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] Olive Optimizer execution complete.');
  });
});
