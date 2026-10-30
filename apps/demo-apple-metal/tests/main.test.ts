// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initAppleMetalDemo } from '../src/main.js';

describe('demo-apple-metal', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run Apple Metal demo', async () => {
    document.body.innerHTML = '';
    initAppleMetalDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initAppleMetalDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing Apple Metal...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('Apple Metal engine loaded.');
  });
});
