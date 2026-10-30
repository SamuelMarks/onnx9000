// @vitest-environment jsdom

import * as backendWeb from '@onnx9000/backend-web';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initProgressiveLoadingDemo } from '../src/main.js';

describe('demo-progressive-loading', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="modelUrl" value="https://example.com/model.onnx"/><button id="loadBtn"></button><button id="runBtn"></button><div id="output"></div>';
  });

  it('should load progressive session, run inference, and handle errors', async () => {
    document.body.innerHTML = '';
    initProgressiveLoadingDemo();

    document.body.innerHTML =
      '<input id="modelUrl" value="https://example.com/model.onnx"/><button id="loadBtn"></button><button id="runBtn"></button><div id="output"></div>';
    initProgressiveLoadingDemo();

    const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement;
    const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    runBtn.click();

    const mockRun = vi.fn().mockResolvedValueOnce({ logits: [1, 2] });
    const loadSpy = vi.spyOn(backendWeb, 'loadProgressive').mockResolvedValueOnce({
      run: mockRun,
    } as never as backendWeb.ProgressiveSession);

    loadBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Session initialized!');

    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Success! Progressively loaded weights');

    mockRun.mockRejectedValueOnce(new Error('Stream error'));
    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error during inference: Stream error');

    mockRun.mockRejectedValueOnce('Raw stream error');
    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error during inference: Raw stream error');

    loadSpy.mockRejectedValueOnce(new Error('HTTP 404'));
    loadBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error: HTTP 404');

    loadSpy.mockRejectedValueOnce('Raw HTTP error');
    loadBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(out.textContent).toContain('Error: Raw HTTP error');
    loadSpy.mockRestore();
  });
});
