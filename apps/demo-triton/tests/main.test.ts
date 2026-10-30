// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initTritonDemo } from '../src/main.js';

describe('demo-triton', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="generate-btn"></button><div id="output"></div>';
  });

  it('should generate Triton kernel code on button click', () => {
    document.body.innerHTML = '';
    initTritonDemo();

    document.body.innerHTML = '<button id="generate-btn"></button><div id="output"></div>';
    initTritonDemo();

    const btn = document.getElementById('generate-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('@triton.jit');
  });
});
