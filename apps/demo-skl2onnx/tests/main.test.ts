// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initSkl2OnnxDemo } from '../src/main.js';

describe('demo-skl2onnx', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
  });

  it('should initialize and run SKL2ONNX demo', async () => {
    document.body.innerHTML = '';
    initSkl2OnnxDemo();

    document.body.innerHTML = '<button id="btn-convert"></button>';
    initSkl2OnnxDemo();
    document.getElementById('btn-convert')?.click();

    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
    initSkl2OnnxDemo();

    const btn = document.getElementById('btn-convert') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Parsing Scikit-LearnScikit-Learn structure...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] SKL2ONNX conversion complete.');
  });
});
