// @vitest-environment jsdom

import * as transformersMod from '@onnx9000/transformers';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initTransformersDemo } from '../src/main.js';

describe('demo-transformers', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="transformers-output"></div>';
  });

  it('should execute sentiment-analysis pipeline and handle errors', async () => {
    document.body.innerHTML = '';
    initTransformersDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="transformers-output"></div>';
    initTransformersDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('transformers-output') as HTMLElement;

    btn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Success! Transformers.js pipeline ran successfully.');

    const spy = vi
      .spyOn(transformersMod, 'pipeline')
      .mockRejectedValueOnce(new Error('Model error'));
    btn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Error: Model error');

    spy.mockRejectedValueOnce('raw error');
    btn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Error: raw error');
    spy.mockRestore();
  });
});
