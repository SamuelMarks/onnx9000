import { describe, it, expect, vi } from 'vitest';
import { initMmdnnDemo } from '../src/main.js';

vi.mock('@onnx9000/converters', () => ({
  convert: vi.fn().mockResolvedValue('text result'),
}));
vi.mock('@onnx9000/core', () => ({
  serializeModelProto: vi.fn().mockResolvedValue(new Uint8Array()),
}));

global.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/5678');
global.URL.revokeObjectURL = vi.fn();
// Mock WebGL error logging
console.error = vi.fn();
window.alert = vi.fn();

describe('demo-mmdnn', () => {
  it('should initialize and run conversion', async () => {
    document.body.innerHTML = `
      <select id="src-framework"><option value="keras">keras</option></select>
      <select id="dst-framework"><option value="onnx">onnx</option></select>
      <div id="drop-zone"></div>
      <input id="file-input" type="file" />
      <p id="drop-hint"></p>
      <div id="files-list"></div>
      <button id="btn-convert"></button>
      <button id="btn-download"></button>
      <div id="logs"></div>
    `;
    initMmdnnDemo();

    // push a file to bypass validation
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.h5')] });
    fileInput.dispatchEvent(new Event('change'));

    document.getElementById('btn-convert')?.click();
    await new Promise((r) => setTimeout(r, 50));

    expect(document.getElementById('logs')?.textContent).toContain('Conversion complete!');
  });
});
