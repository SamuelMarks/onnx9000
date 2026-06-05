import { describe, it, expect, vi } from 'vitest';
import { initOnnx2GgufUI } from '../src/main.js';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({ name: 'mock' }),
}));
vi.mock('@onnx9000/onnx2gguf', () => ({
  extractMetadata: vi.fn().mockReturnValue({}),
  extractTokenizerMetadata: vi.fn().mockReturnValue({}),
  inferArchitecture: vi.fn().mockReturnValue('llama'),
  compileGGUF: vi.fn().mockReturnValue(new Uint8Array(10)),
}));
global.alert = vi.fn();
global.URL.createObjectURL = vi.fn().mockReturnValue('blob');
global.URL.revokeObjectURL = vi.fn();

describe('onnx2gguf-ui', () => {
  it('should initialize and convert', async () => {
    document.body.innerHTML = `
      <div id="dropzone"></div>
      <input id="fileInput" type="file" />
      <tbody id="metaTableBody"></tbody>
      <button id="convertBtn"></button>
      <div id="status"></div>
      <div id="warning"></div>
      <select id="quantTarget"><option value="f16">f16</option></select>
    `;
    initOnnx2GgufUI();

    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.onnx')] });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 10));
    document.getElementById('convertBtn')?.click();
    await new Promise((r) => setTimeout(r, 10));

    expect(document.getElementById('status')?.textContent).toContain('Downloaded');
  });
});
