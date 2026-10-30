// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initKeras2OnnxDemo } from '../src/main.js';

describe('demo-keras2onnx', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
  });

  it('should initialize and run Keras2ONNX demo', async () => {
    document.body.innerHTML = '';
    initKeras2OnnxDemo();

    document.body.innerHTML = '<button id="btn-convert"></button>';
    initKeras2OnnxDemo();
    document.getElementById('btn-convert')?.click();

    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
    initKeras2OnnxDemo();

    const btn = document.getElementById('btn-convert') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Parsing KerasKeras structure...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] Keras2ONNX conversion complete.');
  });
});
