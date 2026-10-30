import * as coreMod from '@onnx9000/core';
import * as ggufMod from '@onnx9000/onnx2gguf';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initOnnx2GgufUI } from '../src/main';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({ name: 'test_model' }),
}));

vi.mock('@onnx9000/onnx2gguf', () => ({
  extractMetadata: vi.fn().mockReturnValue({
    'llama.vocab_size': 32000,
    'custom.bool': true,
    'custom.arr': [1, 2],
  }),
  extractTokenizerMetadata: vi.fn().mockReturnValue({ 'tokenizer.model': 'llama' }),
  inferArchitecture: vi.fn().mockReturnValue('llama'),
  compileGGUF: vi.fn().mockReturnValue(new Uint8Array([1, 2, 3]).buffer),
}));

describe('initOnnx2GgufUI', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<div id="dropzone"></div><input type="file" id="fileInput" /><table><tbody id="metaTableBody"></tbody></table><button id="convertBtn"></button><div id="status"></div><div id="warning"></div><select id="quantTarget"><option value="f16">f16</option></select>';
    vi.clearAllMocks();
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock');
    global.URL.revokeObjectURL = vi.fn();
    (global as typeof globalThis & { performance: unknown }).performance = {
      now: () => Date.now(),
    };
  });

  it('should initialize successfully', () => {
    expect(() => initOnnx2GgufUI()).not.toThrow();
  });

  it('should handle missing elements', () => {
    document.body.innerHTML = '';
    expect(() => initOnnx2GgufUI()).not.toThrow();
  });

  it('should handle file drops (onnx)', async () => {
    initOnnx2GgufUI();
    const dropzone = document.getElementById('dropzone');
    dropzone?.dispatchEvent(new Event('dragover'));
    expect(dropzone?.classList.contains('dragover')).toBe(true);
    dropzone?.dispatchEvent(new Event('dragleave'));
    expect(dropzone?.classList.contains('dragover')).toBe(false);

    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    const dropEvent = Object.assign(new Event('drop'), {
      dataTransfer: { files: [mockFile] },
    });
    dropzone?.dispatchEvent(dropEvent);

    await new Promise((r) => setTimeout(r, 0));
    const status = document.getElementById('status');
    expect(status?.textContent).toContain('Ready for conversion');
  });

  it('should handle file input change', async () => {
    initOnnx2GgufUI();
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    const mockFile = new File(['mock'], 'tokenizer.json', {
      type: 'application/json',
    });
    Object.defineProperty(fileInput, 'files', { value: [mockFile] });

    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 0));
    const status = document.getElementById('status');
    expect(status?.textContent).toContain('Please provide an .onnx file');
  });

  it('should compile and download model', async () => {
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    Object.defineProperty(navigator, 'deviceMemory', { value: 4, configurable: true });
    initOnnx2GgufUI();
    expect(document.getElementById('warning')?.textContent).toContain('less than 8GB');

    const convertBtn = document.getElementById('convertBtn');
    convertBtn?.click(); // before loading model -> alert
    expect(alertSpy).toHaveBeenCalled();

    const dropzone = document.getElementById('dropzone');
    dropzone?.click();
    dropzone?.dispatchEvent(new Event('drop')); // without dataTransfer

    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: null, configurable: true });
    fileInput.dispatchEvent(new Event('change'));

    vi.mocked(coreMod.load).mockResolvedValueOnce({ name: '' } as never);
    vi.mocked(ggufMod.extractMetadata).mockReturnValueOnce({
      'custom.bool': true,
      'custom.arr': [1, 2],
      'custom.num': 10,
    });

    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    const ignoredFile = new File(['mock'], 'ignored.txt', { type: 'text/plain' });
    const dropEvent = Object.assign(new Event('drop'), {
      dataTransfer: { files: [mockFile, ignoredFile] },
    });
    dropzone?.dispatchEvent(dropEvent);

    await new Promise((r) => setTimeout(r, 0));

    // Trigger change on all inputs in metaTableBody (string, number, boolean, array)
    const inputs = document.querySelectorAll('#metaTableBody input');
    inputs.forEach((inp) => {
      inp.dispatchEvent(new Event('change'));
    });

    convertBtn?.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(global.URL.createObjectURL).toHaveBeenCalled();

    // showSaveFilePicker branch
    const writeMock = vi.fn().mockResolvedValue(undefined);
    const closeMock = vi.fn().mockResolvedValue(undefined);
    (window as unknown as { showSaveFilePicker: unknown }).showSaveFilePicker = vi
      .fn()
      .mockResolvedValue({
        createWritable: () => Promise.resolve({ write: writeMock, close: closeMock }),
      });
    convertBtn?.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(writeMock).toHaveBeenCalled();
    delete (window as unknown as { showSaveFilePicker?: unknown }).showSaveFilePicker;

    // Error with 'memory' and non-Error throw
    vi.mocked(ggufMod.compileGGUF).mockImplementationOnce(() => {
      throw new Error('out of memory');
    });
    convertBtn?.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(document.getElementById('status')?.textContent).toContain('out of memory');

    vi.mocked(ggufMod.compileGGUF).mockImplementationOnce(() => {
      throw 'raw fail';
    });
    convertBtn?.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(document.getElementById('status')?.textContent).toContain('raw fail');

    alertSpy.mockRestore();
  });
});
