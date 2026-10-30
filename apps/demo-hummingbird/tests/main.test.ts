// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initHummingbirdDemo } from '../src/main.js';

describe('demo-hummingbird', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="transpile-btn"></button><div id="hb-output"></div>';
  });

  it('should transpile tree ensemble and update output', async () => {
    document.body.innerHTML = '';
    initHummingbirdDemo();

    document.body.innerHTML = '<button id="transpile-btn"></button><div id="hb-output"></div>';
    initHummingbirdDemo();

    const btn = document.getElementById('transpile-btn') as HTMLButtonElement;
    const out = document.getElementById('hb-output') as HTMLElement;

    btn.click();
    await new Promise((r) => setTimeout(r, 150));

    expect(out.innerText).toContain('Transpilation successful!');
    expect(btn.disabled).toBe(true);
  });
});
