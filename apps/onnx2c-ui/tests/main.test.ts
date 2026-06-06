import { beforeEach, describe, expect, it, vi } from 'vitest';

(global as any).Worker = class Worker {
  onmessage: any;
  postMessage(_msg: any) {
    if (this.onmessage) {
      this.onmessage({
        data: {
          header: 'mock header',
          source: 'mock source',
          summary: 'mock summary',
          arenaSize: 500,
        },
      });
    }
  }
};
global.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock');
global.URL.revokeObjectURL = vi.fn();

import * as monaco from 'monaco-editor';
import { initOnnx2cUI } from '../src/main';

vi.mock('monaco-editor', () => ({
  editor: {
    create: vi.fn().mockReturnValue({
      setValue: vi.fn(),
      getValue: vi.fn().mockReturnValue('mock code'),
    }),
  },
}));

describe('initOnnx2cUI', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<div id="monaco-root"></div><div id="dropzone"></div><input type="file" id="file-input" /><button id="btn-compile"></button><button id="btn-download"></button><select id="target-board"><option value="1000">1000</option></select><select id="target-arch"><option value="c">c</option></select><input type="checkbox" id="opt-cpp" /><input type="checkbox" id="opt-math" /><input type="checkbox" id="opt-unroll" /><div id="controls"></div><div class="tab" data-target="header">Header</div><div class="tab" data-target="source">Source</div>';
    vi.clearAllMocks();
  });

  it('should initialize successfully', () => {
    initOnnx2cUI();
    expect(monaco.editor.create).toHaveBeenCalled();
  });

  it('should handle missing elements safely', () => {
    document.body.innerHTML = '';
    expect(() => initOnnx2cUI()).not.toThrow();
  });

  it('should handle tab clicks', () => {
    initOnnx2cUI();
    const tabSource = document.querySelector('.tab[data-target="source"]') as HTMLElement;
    tabSource.click();
    expect(tabSource.classList.contains('active')).toBe(true);
  });

  it('should handle file drops', async () => {
    initOnnx2cUI();
    const dropzone = document.getElementById('dropzone')!;
    dropzone.dispatchEvent(new Event('dragover'));
    expect(dropzone.classList.contains('hover')).toBe(true);
    dropzone.dispatchEvent(new Event('dragleave'));
    expect(dropzone.classList.contains('hover')).toBe(false);

    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropzone.dispatchEvent(dropEvent);

    await new Promise((r) => setTimeout(r, 0));
    expect(dropzone.innerHTML).toContain('test.onnx');
  });

  it('should handle file input changes', async () => {
    initOnnx2cUI();
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    Object.defineProperty(fileInput, 'files', { value: [mockFile] });

    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 0));

    const dropzone = document.getElementById('dropzone')!;
    expect(dropzone.innerHTML).toContain('test.onnx');
  });

  it('should compile model when compile button is clicked', async () => {
    initOnnx2cUI();
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    Object.defineProperty(fileInput, 'files', { value: [mockFile] });
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 0));

    const compileBtn = document.getElementById('btn-compile')!;
    compileBtn.click();

    expect(compileBtn.innerText).toBe('Compile to C');
  });

  it('should download model when download button is clicked', () => {
    initOnnx2cUI();
    const downloadBtn = document.getElementById('btn-download')!;
    downloadBtn.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
