import { describe, it, expect, vi, beforeEach } from 'vitest';
import { initOpenVinoUI } from '../src/main';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({ name: 'test_model' })
}));

vi.mock('@onnx9000/openvino-exporter', () => ({
  OpenVinoExporter: class {
    export() {
      return { xml: '<xml>', bin: new Uint8Array([1,2,3]) };
    }
  }
}));

vi.mock('jszip', () => {
  return {
    default: class JSZip {
      file() {}
      generateAsync() { return Promise.resolve(new Blob(['mock zip'])); }
    }
  };
});

describe('initOpenVinoUI', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="dropzone"></div><input type="file" id="file-input" /><div id="status"></div><input type="checkbox" id="compressFp16" />';
    vi.clearAllMocks();
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('should initialize successfully', () => {
    expect(() => initOpenVinoUI()).not.toThrow();
  });

  it('should handle missing elements', () => {
    document.body.innerHTML = '';
    expect(() => initOpenVinoUI()).not.toThrow();
  });

  it('should handle drag events', () => {
    initOpenVinoUI();
    const dropzone = document.getElementById('dropzone')!;
    dropzone.dispatchEvent(new Event('dragenter'));
    expect(dropzone.classList.contains('active')).toBe(true);
    dropzone.dispatchEvent(new Event('dragleave'));
    expect(dropzone.classList.contains('active')).toBe(false);
  });

  it('should handle file drops (not onnx)', async () => {
    initOpenVinoUI();
    const dropzone = document.getElementById('dropzone')!;
    const mockFile = new File(['mock'], 'test.txt', { type: 'text/plain' });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropzone.dispatchEvent(dropEvent);

    await new Promise(r => setTimeout(r, 0));
    const status = document.getElementById('status')!;
    expect(status.innerHTML).toContain('Please drop an .onnx file.');
  });

  it('should handle file drops (onnx)', async () => {
    initOpenVinoUI();
    const dropzone = document.getElementById('dropzone')!;
    const mockFile = new File(['mock'], 'test.onnx', { type: 'application/octet-stream' });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropzone.dispatchEvent(dropEvent);

    await new Promise(r => setTimeout(r, 100)); // allow setTimeout in main.ts
    const status = document.getElementById('status')!;
    expect(status.innerHTML).toContain('Success! Downloaded test_openvino.zip');
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });

  it('should handle file input change', async () => {
    initOpenVinoUI();
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const mockFile = new File(['mock'], 'test.onnx', { type: 'application/octet-stream' });
    Object.defineProperty(fileInput, 'files', { value: [mockFile] });
    
    fileInput.dispatchEvent(new Event('change'));
    await new Promise(r => setTimeout(r, 100));
    const status = document.getElementById('status')!;
    expect(status.innerHTML).toContain('Success! Downloaded test_openvino.zip');
  });

  it('should trigger click on dropzone click', () => {
    initOpenVinoUI();
    const dropzone = document.getElementById('dropzone')!;
    const fileInput = document.getElementById('file-input')!;
    const spy = vi.spyOn(fileInput, 'click');
    dropzone.click();
    expect(spy).toHaveBeenCalled();
  });
});
