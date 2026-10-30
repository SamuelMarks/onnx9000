// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initWebGpuDemo } from '../src/main.js';

describe('demo-webgpu', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run WebGPU demo', async () => {
    document.body.innerHTML = '';
    initWebGpuDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initWebGpuDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing WebGPU...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('WebGPU engine loaded.');
  });
});
