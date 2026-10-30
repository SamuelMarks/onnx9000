// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initTvmDemo } from '../src/main.js';

describe('demo-tvm', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
  });

  it('should convert graph and output Relay IR text', () => {
    document.body.innerHTML = '';
    initTvmDemo();

    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
    initTvmDemo();

    const btn = document.getElementById('convert-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('nn.relu(%input1)');
  });
});
