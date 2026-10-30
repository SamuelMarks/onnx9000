// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initAgentDemo } from '../src/main.js';

describe('demo-agent', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<textarea id="prompt"></textarea><button id="runBtn"></button><div id="output"></div>';
  });

  it('should handle missing DOM elements gracefully', () => {
    document.body.innerHTML = '';
    initAgentDemo();
    expect(document.body.innerHTML).toBe('');
  });

  it('should run agent workflow and populate final answer', async () => {
    initAgentDemo();

    const btn = document.getElementById('runBtn') as HTMLButtonElement;
    const prompt = document.getElementById('prompt') as HTMLTextAreaElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toBeUndefined();

    prompt.value = 'test';
    btn.click();

    await new Promise((r) => setTimeout(r, 350));
    expect(out.innerText).toContain('[Agent] Final Answer: 55');
    expect(btn.disabled).toBe(false);
  });
});
