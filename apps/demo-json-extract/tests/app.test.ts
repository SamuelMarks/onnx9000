// @vitest-environment jsdom
import * as coreMod from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initJsonExtractDemo } from '../app.js';

describe('demo-json-extract', () => {
  beforeEach(() => {
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/1234');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('should extract JSON, handle drag/drop, buffers, bigints, and errors', async () => {
    document.body.innerHTML = '';
    initJsonExtractDemo();

    document.body.innerHTML = `
      <div id="drop-zone"></div>
      <input type="file" id="file-input" />
      <button id="browse-btn"></button>
      <div id="status-panel" class="hidden"></div>
      <div id="result-panel" class="hidden"></div>
      <div id="status-text"></div>
      <div id="progress-bar"></div>
      <div id="error-box" class="hidden"></div>
      <button id="download-btn"></button>
      <div id="stats-text"></div>
    `;
    initJsonExtractDemo();

    const dropZone = document.getElementById('drop-zone') as HTMLElement;
    const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;

    browseBtn.click();
    downloadBtn.click();
    fileInput.dispatchEvent(new Event('change'));

    dropZone.dispatchEvent(new Event('dragover', { cancelable: true }));
    expect(dropZone.classList.contains('dragover')).toBe(true);
    dropZone.dispatchEvent(new Event('dragleave'));
    expect(dropZone.classList.contains('dragover')).toBe(false);
    dropZone.dispatchEvent(new Event('drop', { cancelable: true }));

    const badFile = new File(['bad'], 'model.txt');
    const dropEvt = new Event('drop', { cancelable: true }) as DragEvent;
    Object.defineProperty(dropEvt, 'dataTransfer', {
      value: { files: [badFile] },
    });
    dropZone.dispatchEvent(dropEvt);
    expect(document.getElementById('error-box')?.textContent).toBe(
      'Please provide a valid .onnx file.',
    );

    const loadSpy = vi.spyOn(coreMod, 'load').mockResolvedValueOnce({
      nodes: [{ opType: 'Identity' }],
      tensors: { w: { data: new Float32Array([1, 2]), big: 99n } },
    } as never as coreMod.Graph);

    const testFile = new File(['123'], 'model.onnx');
    testFile.arrayBuffer = async () => new ArrayBuffer(0);
    Object.defineProperty(fileInput, 'files', { value: [testFile], configurable: true });

    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 50));

    expect(document.getElementById('status-text')?.textContent).toBe('Done!');
    expect(document.getElementById('stats-text')?.innerHTML).toContain('model.onnx');

    downloadBtn.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();

    loadSpy.mockRejectedValueOnce(new Error('Corrupt ONNX'));
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 30));
    expect(document.getElementById('error-box')?.textContent).toBe('Corrupt ONNX');

    loadSpy.mockRejectedValueOnce('Raw corrupt');
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 30));
    expect(document.getElementById('error-box')?.textContent).toBe('Raw corrupt');

    loadSpy.mockRestore();
  });
});
