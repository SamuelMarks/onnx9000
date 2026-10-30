// @vitest-environment jsdom

import { mmdnn } from '@onnx9000/converters';
import { Graph } from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initExtendedConvertersDemo } from '../src/main.js';

describe('demo-extended-converters', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input type="file" id="fileInput"/><select id="srcFramework"><option value="caffe">caffe</option></select><select id="dstFramework"><option value="onnx">onnx</option></select><button id="btnConvert"></button><div id="output"></div>';
  });

  it('should convert selected files for string, graph, and error outcomes', async () => {
    document.body.innerHTML = '';
    initExtendedConvertersDemo();

    document.body.innerHTML =
      '<input type="file" id="fileInput"/><select id="srcFramework"><option value="caffe">caffe</option></select><select id="dstFramework"><option value="onnx">onnx</option></select><button id="btnConvert"></button><div id="output"></div>';
    initExtendedConvertersDemo();

    const btnConvert = document.getElementById('btnConvert') as HTMLButtonElement;
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    const out = document.getElementById('output') as HTMLDivElement;

    btnConvert.click();
    expect(out.textContent).toBe('Please select one or more files to convert.');

    const f = new File(['data'], 'test.prototxt');
    Object.defineProperty(fileInput, 'files', { value: [f], configurable: true });

    const convertSpy = vi.spyOn(mmdnn, 'convert').mockResolvedValueOnce('code_payload');
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Result Type: String Payload');

    convertSpy.mockResolvedValueOnce(new Graph('test_g'));
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Result Type: Graph Object');

    convertSpy.mockRejectedValueOnce(new Error('Invalid prototxt'));
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error during conversion: Invalid prototxt');

    convertSpy.mockRejectedValueOnce('Raw conv error');
    btnConvert.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error during conversion: Raw conv error');
    convertSpy.mockRestore();
  });
});
