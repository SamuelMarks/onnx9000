// @vitest-environment jsdom
import * as convertersMod from '@onnx9000/converters';
import * as coreMod from '@onnx9000/core';
import { Graph } from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initMmdnnDemo } from '../src/main.js';

describe('demo-mmdnn', () => {
  beforeEach(() => {
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/mmdnn');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('should validate all frameworks, convert models, and handle WebGL and download flows', async () => {
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => undefined);
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    document.body.innerHTML = '';
    initMmdnnDemo();

    const frameworks = [
      'caffe',
      'mxnet',
      'cntk',
      'darknet',
      'ncnn',
      'paddle',
      'keras',
      'coreml',
      'jax',
      'flax',
      'h2o',
      'libsvm',
      'sklearn',
      'xgboost',
      'catboost',
      'safetensors',
      'lightgbm',
      'pyspark',
      'unknown',
    ];
    const targets = ['onnx', 'pytorch_code', 'tfjs', 'other'];

    const domHtml = `
      <select id="src-framework">${frameworks.map((f) => `<option value="${f}">${f}</option>`).join('')}</select>
      <select id="dst-framework">${targets.map((t) => `<option value="${t}">${t}</option>`).join('')}</select>
      <div id="drop-zone"></div>
      <input type="file" id="file-input" />
      <p id="drop-hint"></p>
      <div id="files-list"></div>
      <button id="btn-convert"></button>
      <button id="btn-download"></button>
      <div id="logs"></div>
      <div id="graph-preview"></div>
    `;
    document.body.innerHTML = domHtml;

    // First init with both getContext calls returning null (WebGL unsupported branch)
    const ctxSpy = vi
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValueOnce(null)
      .mockReturnValueOnce(null)
      .mockReturnValueOnce(null)
      .mockReturnValueOnce({} as never as RenderingContext);
    initMmdnnDemo();
    document.body.innerHTML = domHtml;
    // Second init with webgl fallback context succeeding
    initMmdnnDemo();
    ctxSpy.mockRestore();

    const srcSelect = document.getElementById('src-framework') as HTMLSelectElement;
    const dstSelect = document.getElementById('dst-framework') as HTMLSelectElement;
    const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const filesList = document.getElementById('files-list') as HTMLDivElement;
    const btnConvert = document.getElementById('btn-convert') as HTMLButtonElement;
    const btnDownload = document.getElementById('btn-download') as HTMLButtonElement;
    const logs = document.getElementById('logs') as HTMLDivElement;

    btnDownload.click();
    dropZone.click();
    dropZone.dispatchEvent(new Event('dragover', { cancelable: true }));
    expect(dropZone.classList.contains('dragover')).toBe(true);
    dropZone.dispatchEvent(new Event('dragleave'));
    expect(dropZone.classList.contains('dragover')).toBe(false);
    dropZone.dispatchEvent(new Event('drop', { cancelable: true }));

    fileInput.dispatchEvent(new Event('change'));

    const allTestFiles = [
      new File(['a'], 'deploy.prototxt'),
      new File(['a'], 'weights.caffemodel'),
      new File(['a'], 'net-symbol.json'),
      new File(['a'], 'net.params'),
      new File(['a'], 'cntk.model'),
      new File(['a'], 'yolo.cfg'),
      new File(['a'], 'yolo.weights'),
      new File(['a'], 'mobilenet.param'),
      new File(['a'], 'mobilenet.bin'),
      new File(['a'], '__model__'),
      new File(['a'], 'model.pdmodel'),
      new File(['a'], 'keras.h5'),
      new File(['a'], 'keras.keras'),
      new File(['a'], 'core.mlmodel'),
      new File(['a'], 'flax.msgpack'),
      new File(['a'], 'mojo.zip'),
      new File(['a'], 'data.svm'),
      new File(['a'], 'data.txt'),
      new File(['a'], 'sk.joblib'),
      new File(['a'], 'model.safetensors'),
      new File(['a'], 'dump.json'),
    ];

    const dropEvt = new Event('drop', { cancelable: true }) as DragEvent;
    Object.defineProperty(dropEvt, 'dataTransfer', {
      value: { files: allTestFiles },
    });
    dropZone.dispatchEvent(dropEvt);

    for (const fw of frameworks) {
      srcSelect.value = fw;
      srcSelect.dispatchEvent(new Event('change'));
    }

    Object.defineProperty(fileInput, 'files', {
      value: [new File(['x'], 'extra.json')],
      configurable: true,
    });
    fileInput.dispatchEvent(new Event('change'));
    const removeButtons = filesList.querySelectorAll('button');
    removeButtons[removeButtons.length - 1]?.click();

    srcSelect.value = 'caffe';
    srcSelect.dispatchEvent(new Event('change'));

    const convertSpy = vi.spyOn(convertersMod, 'convert').mockResolvedValueOnce(new Graph('out_g'));
    const serializeSpy = vi
      .spyOn(coreMod, 'serializeModelProto')
      .mockResolvedValueOnce(new Uint8Array([1, 2, 3]));

    dstSelect.value = 'onnx';
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(logs.textContent).toContain('Conversion complete!');

    btnDownload.click();

    dstSelect.value = 'pytorch_code';
    convertSpy.mockResolvedValueOnce('import torch');
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(global.URL.revokeObjectURL).toHaveBeenCalled();

    dstSelect.value = 'other';
    convertSpy.mockResolvedValueOnce({ format: 'json' } as never as Graph);
    document.getElementById('graph-preview')?.remove();
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 40));

    convertSpy.mockRejectedValueOnce(new Error('MMDNN failure'));
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(logs.textContent).toContain('Conversion failed: MMDNN failure');

    convertSpy.mockRejectedValueOnce('Raw MMDNN failure');
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(logs.textContent).toContain('Conversion failed: Raw MMDNN failure');

    convertSpy.mockRestore();
    serializeSpy.mockRestore();
    alertSpy.mockRestore();
    errSpy.mockRestore();
  });
});
