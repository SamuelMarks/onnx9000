// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initJaxDemo } from '../src/main.js';

describe('demo-jax', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="jax-output"></div>';
  });

  it('should parse ClosedJaxpr and report mapped equations', () => {
    document.body.innerHTML = '';
    initJaxDemo();

    document.body.innerHTML = '<button id="convert-btn"></button><div id="jax-output"></div>';
    initJaxDemo();

    const btn = document.getElementById('convert-btn') as HTMLButtonElement;
    const out = document.getElementById('jax-output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Success! JAX & Flax graphs can be transpiled natively in JS.');
    expect(btn.disabled).toBe(false);
  });
});
