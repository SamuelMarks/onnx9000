// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initTensorrtDemo } from '../src/main.js';

describe('demo-tensorrt', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
  });

  it('should run flow and output TensorRT script', () => {
    document.body.innerHTML = '';
    initTensorrtDemo();

    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
    initTensorrtDemo();

    const btn = document.getElementById('convert-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('import tensorrt as trt');
  });
});
