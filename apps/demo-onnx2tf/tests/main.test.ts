import { describe, it, expect, vi } from 'vitest';
import { initOnnx2TfDemo } from '../src/main.js';

describe('demo-onnx2tf', () => {
  it('should run conversion', () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button id="convertBtn"></button>
      <button id="resetBtn"></button>
      <div id="output"></div>
      <input id="modelPath" value="model.onnx" />
      <input id="outputPath" value="model.tflite" />
      <input type="checkbox" id="int8Quant" checked />
    `;
    initOnnx2TfDemo();
    document.getElementById('convertBtn')?.click();
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('onnx2tf conversion complete');

    document.getElementById('resetBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Waiting to convert');
    vi.useRealTimers();
  });
});
