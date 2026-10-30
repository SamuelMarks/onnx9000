// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initRocmDemo } from '../src/main.js';

describe('demo-rocm', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run ROCm demo', async () => {
    document.body.innerHTML = '';
    initRocmDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initRocmDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing ROCm...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('ROCm engine loaded.');
  });
});
