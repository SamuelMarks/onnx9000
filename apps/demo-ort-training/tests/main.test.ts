// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOrtTrainingDemo } from '../src/main.js';

describe('demo-ort-training', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
  });

  it('should initialize and run ORT training demo', async () => {
    document.body.innerHTML = '';
    initOrtTrainingDemo();

    document.body.innerHTML = '<button id="btn-run"></button>';
    initOrtTrainingDemo();
    document.getElementById('btn-run')?.click();

    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOrtTrainingDemo();

    const btn = document.getElementById('btn-run') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Running...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] ORT Training execution complete.');
  });
});
