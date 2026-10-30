// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initTritonServerDemo } from '../src/main.js';

describe('demo-triton-server', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
  });

  it('should initialize and run Triton server demo', async () => {
    document.body.innerHTML = '';
    initTritonServerDemo();

    document.body.innerHTML = '<button id="btn-run"></button>';
    initTritonServerDemo();
    document.getElementById('btn-run')?.click();

    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initTritonServerDemo();

    const btn = document.getElementById('btn-run') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Running...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] Triton Server execution complete.');
  });
});
