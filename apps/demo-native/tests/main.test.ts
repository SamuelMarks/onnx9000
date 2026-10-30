// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initNativeCpuDemo } from '../src/main.js';

describe('demo-native', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run Native CPU demo', async () => {
    document.body.innerHTML = '';
    initNativeCpuDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initNativeCpuDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing Native CPU...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('Native CPU engine loaded.');
  });
});
