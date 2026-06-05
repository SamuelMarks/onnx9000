import { describe, it, expect, vi } from 'vitest';

vi.mock('@onnx9000/core', () => ({
  parseModelProto: vi.fn().mockResolvedValue({}),
  BufferReader: class {},
}));
vi.mock('@onnx9000/tflite-exporter', () => ({
  compileGraphToTFLite: vi.fn().mockReturnValue(0),
  TFLiteExporter: class {
    builder = { startVector: vi.fn(), addOffset: vi.fn(), endVector: vi.fn().mockReturnValue(0) };
    finish = vi.fn().mockReturnValue(new Uint8Array());
  },
}));

global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
global.URL.revokeObjectURL = vi.fn();

describe('demo-tflite-converter', () => {
  it('should run converter', async () => {
    document.body.innerHTML = `
      <div id="drop-zone"></div>
      <input id="file-input" type="file" />
      <button id="browse-btn"></button>
      <div id="status-panel" class="hidden"></div>
      <div id="result-panel" class="hidden"></div>
      <p id="status-text"></p>
      <div id="progress-bar"></div>
      <div id="error-box" class="hidden"></div>
      <button id="download-btn"></button>
      <p id="stats-text"></p>
      <input id="opt-edgetpu" type="checkbox" />
      <input id="quant-fp16" type="checkbox" />
      <input id="quant-int8" type="checkbox" />
      <input id="open-netron" type="checkbox" />
    `;
    await import('../app.js');

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.onnx')] });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 50));
    expect(document.getElementById('status-text')?.textContent).toContain('Done!');

    document.getElementById('download-btn')?.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
