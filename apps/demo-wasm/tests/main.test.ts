// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initWasmDemo } from '../src/main.js';

describe('demo-wasm', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run WASM demo', async () => {
    document.body.innerHTML = '';
    initWasmDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initWasmDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing WASM Compiler...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('WASM engine loaded.');
  });
});
