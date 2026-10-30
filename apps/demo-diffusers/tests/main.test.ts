// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initDiffusersDemo } from '../src/main.js';

describe('demo-diffusers', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should initialize Diffusers pipeline and generate mock tensor', () => {
    document.body.innerHTML = '';
    initDiffusersDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initDiffusersDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Image tensor generated successfully [1, 3, 512, 512]');
  });
});
