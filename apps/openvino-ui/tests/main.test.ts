import { describe, it, expect, vi } from 'vitest';
import { initOpenVinoUI } from '../src/main.js';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({}),
}));
vi.mock('@onnx9000/openvino-exporter', () => ({
  OpenVinoExporter: class {
    export() {
      return { xml: 'x', bin: new Uint8Array() };
    }
  },
}));

global.URL.createObjectURL = vi.fn().mockReturnValue('blob');
global.URL.revokeObjectURL = vi.fn();

describe('openvino-ui', () => {
  it('should initialize and convert', async () => {
    document.body.innerHTML = `
      <div id="dropzone"></div>
      <input id="file-input" type="file" />
      <div id="status"></div>
      <input id="compressFp16" type="checkbox" />
    `;
    initOpenVinoUI();

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.onnx')] });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 100));
    expect(document.getElementById('status')?.textContent).toContain('Success');
  });
});
