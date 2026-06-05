import { describe, it, expect, vi } from 'vitest';
import { initOnnx2cUI } from '../src/main.js';

vi.mock('monaco-editor', () => ({
  editor: { create: vi.fn().mockReturnValue({ setValue: vi.fn() }) },
}));
(global as any).Worker = class {
  postMessage() {
    if ((this as any).onmessage) {
      (this as any).onmessage({
        data: { header: 'h', source: 's', summary: 'sum', arenaSize: 10 },
      });
    }
  }
};
global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
global.URL.revokeObjectURL = vi.fn();
global.alert = vi.fn();

describe('onnx2c-ui', () => {
  it('should initialize', async () => {
    document.body.innerHTML = `
      <div id="monaco-root"></div>
      <div id="dropzone"></div>
      <input id="file-input" type="file" />
      <button id="btn-compile"></button>
      <button id="btn-download"></button>
      <select id="target-board"><option value="1000">1k</option></select>
      <div id="controls"></div>
      <select id="target-arch"><option value="wasm">wasm</option></select>
      <input id="opt-cpp" type="checkbox" />
      <input id="opt-math" type="checkbox" />
      <input id="opt-unroll" type="checkbox" />
    `;
    initOnnx2cUI();

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.onnx')] });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 10));

    document.getElementById('btn-compile')?.click();
    await new Promise((r) => setTimeout(r, 10));

    document.getElementById('btn-download')?.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
