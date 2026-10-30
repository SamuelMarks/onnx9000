// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initPaddle2OnnxDemo } from '../src/main.js';

describe('demo-paddle2onnx', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
  });

  it('should initialize and run Paddle2ONNX demo', async () => {
    document.body.innerHTML = '';
    initPaddle2OnnxDemo();

    document.body.innerHTML = '<button id="btn-convert"></button>';
    initPaddle2OnnxDemo();
    document.getElementById('btn-convert')?.click();

    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
    initPaddle2OnnxDemo();

    const btn = document.getElementById('btn-convert') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Parsing PaddlePaddle structure...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] Paddle2ONNX conversion complete.');
  });
});
