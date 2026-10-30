// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOnnx2TfDemo } from '../src/main.js';

describe('demo-onnx2tf', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><input type="checkbox" id="int8Quant"/><button id="convertBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should convert with and without INT8 quantization and reset', async () => {
    document.body.innerHTML = '';
    initOnnx2TfDemo();

    document.body.innerHTML =
      '<input id="modelPath"/><input id="outputPath"/><input type="checkbox" id="int8Quant"/><button id="convertBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initOnnx2TfDemo();

    const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const int8Quant = document.getElementById('int8Quant') as HTMLInputElement;
    const out = document.getElementById('output') as HTMLDivElement;

    convertBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('onnx2tf conversion complete.');

    resetBtn.click();
    expect(out.textContent).toContain('Waiting to convert...');

    int8Quant.checked = true;
    convertBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('with INT8 quantization');
  });
});
