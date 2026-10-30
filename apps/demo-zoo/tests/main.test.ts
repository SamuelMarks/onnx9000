// @vitest-environment jsdom

import * as coreMod from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initZooDemo } from '../src/main.js';

describe('demo-zoo', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="fetch-btn"></button><button id="stream-btn"></button><div id="zoo-output"></div><div id="progress-bar"></div>';
  });

  it('should fetch Safetensors header, stream tensors, and handle errors', async () => {
    document.body.innerHTML = '';
    initZooDemo();

    document.body.innerHTML =
      '<button id="fetch-btn"></button><button id="stream-btn"></button><div id="zoo-output"></div><div id="progress-bar"></div>';
    initZooDemo();

    const fetchBtn = document.getElementById('fetch-btn') as HTMLButtonElement;
    const streamBtn = document.getElementById('stream-btn') as HTMLButtonElement;
    const out = document.getElementById('zoo-output') as HTMLElement;

    const headerSpy = vi.spyOn(coreMod, 'fetchSafetensorsHeader').mockResolvedValueOnce({
      headerObj: { __metadata__: { format: 'pt' } } as never as coreMod.SafetensorsHeader,
      headerSize: 128,
    });

    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Format: pt');

    headerSpy.mockResolvedValueOnce({
      headerObj: { __metadata__: { format: '' } } as never as coreMod.SafetensorsHeader,
      headerSize: 64,
    });
    fetchBtn.disabled = false;
    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));

    headerSpy.mockResolvedValueOnce({
      headerObj: { __metadata__: {} } as never as coreMod.SafetensorsHeader,
      headerSize: 64,
    });
    fetchBtn.disabled = false;
    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));

    headerSpy.mockResolvedValueOnce({
      headerObj: {} as never as coreMod.SafetensorsHeader,
      headerSize: 32,
    });
    fetchBtn.disabled = false;
    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));

    headerSpy.mockRejectedValueOnce(new Error('Fetch fail'));
    fetchBtn.disabled = false;
    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Error: Fetch fail');

    headerSpy.mockRejectedValueOnce('Raw fetch fail');
    fetchBtn.disabled = false;
    fetchBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Error: Raw fetch fail');
    headerSpy.mockRestore();

    const loadSpy = vi.spyOn(coreMod, 'loadTensors').mockImplementationOnce(async function* () {
      yield {
        name: 'w1',
        info: { dtype: 'F32', shape: [2, 2], data_offsets: [0, 16] },
        data: new Uint8Array(16),
      };
      yield {
        name: 'w2',
        info: { dtype: 'F32', shape: [2], data_offsets: [16, 24] },
        data: new Uint8Array(8),
      };
    });

    streamBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Success! Progressively loaded weights');

    loadSpy.mockImplementationOnce(async function* () {
      yield {
        name: 'w1',
        info: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
        data: new Uint8Array(8),
      };
    });
    streamBtn.click();
    await new Promise((r) => setTimeout(r, 30));

    loadSpy.mockImplementationOnce(() => ({
      next: () => Promise.reject(new Error('Range error')),
      return: () => Promise.resolve({ value: undefined, done: true }),
      throw: (e: unknown) => Promise.reject(e),
      [Symbol.asyncIterator]() {
        return this;
      },
    }));
    streamBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Error: Range error');

    loadSpy.mockImplementationOnce(() => ({
      next: () => Promise.reject('Raw range error'),
      return: () => Promise.resolve({ value: undefined, done: true }),
      throw: (e: unknown) => Promise.reject(e),
      [Symbol.asyncIterator]() {
        return this;
      },
    }));
    streamBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.innerText).toContain('Error: Raw range error');
    loadSpy.mockRestore();
  });
});
