// @vitest-environment jsdom

import * as coreMod from '@onnx9000/core';
import { Graph } from '@onnx9000/core';
import * as tfliteMod from '@onnx9000/tflite-exporter';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initTfliteConverterDemo } from '../app.js';

describe('demo-tflite-converter', () => {
  beforeEach(() => {
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/tflite');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('should convert ONNX file to TFLite, support quantization/Netron options, and handle errors', async () => {
    document.body.innerHTML = '';
    initTfliteConverterDemo();

    document.body.innerHTML = `
      <div id="drop-zone"></div>
      <input type="file" id="file-input" />
      <button id="browse-btn"></button>
      <div id="status-panel" class="hidden"></div>
      <div id="result-panel" class="hidden"></div>
      <p id="status-text"></p>
      <div id="progress-bar"></div>
      <div id="error-box" class="hidden"></div>
      <button id="download-btn"></button>
      <p id="stats-text"></p>
      <input type="checkbox" id="opt-edgetpu" />
      <input type="checkbox" id="quant-fp16" />
      <input type="checkbox" id="quant-int8" />
      <input type="checkbox" id="open-netron" />
    `;
    initTfliteConverterDemo();

    const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement;
    const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
    const quantFp16 = document.getElementById('quant-fp16') as HTMLInputElement;
    const quantInt8 = document.getElementById('quant-int8') as HTMLInputElement;
    const openNetron = document.getElementById('open-netron') as HTMLInputElement;

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

    const parseSpy = vi.spyOn(coreMod, 'parseModelProto').mockResolvedValue(new Graph('tflite_g'));
    const compileSpy = vi.spyOn(tfliteMod, 'compileGraphToTFLite').mockReturnValue(4);

    quantFp16.checked = true;
    quantInt8.checked = true;

    const goodFile = new File(['onnx'], 'model.onnx');
    goodFile.arrayBuffer = async () => new ArrayBuffer(4);
    Object.defineProperty(fileInput, 'files', { value: [goodFile], configurable: true });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 80));
    expect(document.getElementById('status-text')?.textContent).toBe('Done!');

    downloadBtn.click();
    openNetron.checked = true;
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    downloadBtn.click();
    expect(openSpy).toHaveBeenCalled();

    await new Promise((r) => setTimeout(r, 80));
    expect(global.URL.revokeObjectURL).toHaveBeenCalled();

    parseSpy.mockRejectedValueOnce(new Error('Invalid proto'));
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 40));
    expect(document.getElementById('error-box')?.textContent).toBe('Invalid proto');

    parseSpy.mockRejectedValueOnce('Raw proto error');
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 40));
    expect(document.getElementById('error-box')?.textContent).toBe('Raw proto error');

    openSpy.mockRestore();
    compileSpy.mockRestore();
    parseSpy.mockRestore();
  });
});
