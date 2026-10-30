// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initCudaDemo } from '../src/main.js';

describe('demo-cuda', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run CUDA demo', async () => {
    document.body.innerHTML = '';
    initCudaDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initCudaDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing CUDA...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('CUDA engine loaded.');
  });
});
