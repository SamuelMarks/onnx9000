// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initGenAIDemo } from '../src/main.js';

describe('demo-genai', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize and run GenAI demo', async () => {
    document.body.innerHTML = '';
    initGenAIDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initGenAIDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Initializing GenAI Subsystem...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.innerText).toContain('GenAI models loaded.');
  });
});
