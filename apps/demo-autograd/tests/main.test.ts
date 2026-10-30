// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initAutogradDemo } from '../src/main.js';

describe('demo-autograd', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="grad-btn"></button><div id="autograd-output"></div>';
  });

  it('should build backward graph and update output', async () => {
    document.body.innerHTML = '';
    initAutogradDemo();

    document.body.innerHTML = '<button id="grad-btn"></button><div id="autograd-output"></div>';
    initAutogradDemo();

    const btn = document.getElementById('grad-btn') as HTMLButtonElement;
    const out = document.getElementById('autograd-output') as HTMLElement;

    btn.click();
    await new Promise((r) => setTimeout(r, 150));

    expect(out.innerText).toContain(
      'Success! Augmented ONNX graph now computes forward pass + gradients.',
    );
    expect(btn.disabled).toBe(true);
  });
});
