// @vitest-environment jsdom

import * as coreMod from '@onnx9000/core';
import { Graph, Node } from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initPytorchCodegenDemo } from '../src/main.js';

describe('demo-pytorch-codegen', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<div id="drop-zone"></div><input type="file" id="file-input"/><textarea id="code"></textarea>';
  });

  it('should handle file selection, drag-and-drop, and codegen errors', async () => {
    document.body.innerHTML = '';
    initPytorchCodegenDemo();

    document.body.innerHTML =
      '<div id="drop-zone"></div><input type="file" id="file-input"/><textarea id="code"></textarea>';
    initPytorchCodegenDemo();

    const dropZone = document.getElementById('drop-zone') as HTMLElement;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const codeArea = document.getElementById('code') as HTMLTextAreaElement;

    dropZone.click();
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
    expect(codeArea.value).toContain('# Error: Please provide a valid .onnx file.');

    const g = new Graph('pt_demo');
    g.nodes.push(new Node('Relu', ['x'], ['y'], {}, 'relu_0'));
    const loadSpy = vi.spyOn(coreMod, 'load').mockResolvedValueOnce(g);

    const goodFile = new File(['onnx'], 'model.onnx');
    goodFile.arrayBuffer = async () => new ArrayBuffer(4);
    Object.defineProperty(fileInput, 'files', { value: [goodFile], configurable: true });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 30));
    expect(codeArea.value).toContain('import torch');

    loadSpy.mockRejectedValueOnce(new Error('Parse failed'));
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 30));
    expect(codeArea.value).toContain('Parse failed');

    loadSpy.mockRejectedValueOnce('Raw parse fail');
    fileInput.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 30));
    expect(codeArea.value).toContain('Raw parse fail');
    loadSpy.mockRestore();
  });
});
