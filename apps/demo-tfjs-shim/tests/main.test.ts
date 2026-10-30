// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initTfjsShimDemo } from '../src/main.js';

describe('demo-tfjs-shim', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
  });

  it('should execute tfjs-shim operations inside tf.tidy', () => {
    document.body.innerHTML = '';
    initTfjsShimDemo();

    document.body.innerHTML = '<button id="run-btn"></button>';
    initTfjsShimDemo();
    document.getElementById('run-btn')?.click();

    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initTfjsShimDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Operations completed inside tf.tidy scope.');
  });
});
