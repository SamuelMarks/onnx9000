import { describe, it, expect, vi, beforeEach } from 'vitest';
import { initOnnx2GgufUI } from '../src/main';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({ name: 'test_model' })
}));

vi.mock('@onnx9000/onnx2gguf', () => ({
  extractMetadata: vi.fn().mockReturnValue({ 'llama.vocab_size': 32000 }),
  extractTokenizerMetadata: vi.fn().mockReturnValue({ 'tokenizer.model': 'llama' }),
  inferArchitecture: vi.fn().mockReturnValue('llama'),
  compileGGUF: vi.fn().mockReturnValue(new Uint8Array([1,2,3]).buffer)
}));

describe('initOnnx2GgufUI', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="dropzone"></div><input type="file" id="fileInput" /><table><tbody id="metaTableBody"></tbody></table><button id="convertBtn"></button><div id="status"></div><div id="warning"></div><select id="quantTarget"><option value="f16">f16</option></select>';
    vi.clearAllMocks();
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock');
    global.URL.revokeObjectURL = vi.fn();
    (global as any).performance = { now: () => Date.now() };
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
    const dropzone = document.getElementById('dropzone')!;
    dropzone.dispatchEvent(new Event('dragover'));
    expect(dropzone.classList.contains('dragover')).toBe(true);
    dropzone.dispatchEvent(new Event('dragleave'));
    expect(dropzone.classList.contains('dragover')).toBe(false);

    const mockFile = new File(['mock'], 'test.onnx', { type: 'application/octet-stream' });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropzone.dispatchEvent(dropEvent);

    await new Promise(r => setTimeout(r, 0));
    const status = document.getElementById('status')!;
    expect(status.textContent).toContain('Ready for conversion');
  });

  it('should handle file input change', async () => {
    initOnnx2GgufUI();
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    const mockFile = new File(['mock'], 'tokenizer.json', { type: 'application/json' });
    Object.defineProperty(fileInput, 'files', { value: [mockFile] });
    
    fileInput.dispatchEvent(new Event('change'));
    await new Promise(r => setTimeout(r, 0));
    const status = document.getElementById('status')!;
    expect(status.textContent).toContain('Please provide an .onnx file');
  });

  it('should compile and download model', async () => {
    initOnnx2GgufUI();
    const dropzone = document.getElementById('dropzone')!;
    const mockFile = new File(['mock'], 'test.onnx', { type: 'application/octet-stream' });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropzone.dispatchEvent(dropEvent);
    
    await new Promise(r => setTimeout(r, 0));

    const convertBtn = document.getElementById('convertBtn')!;
    convertBtn.click();
    
    await new Promise(r => setTimeout(r, 0));
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
