import { describe, it, expect, vi } from 'vitest';
import { initJsonExtractDemo } from '../app.js';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({ nodes: [{ opType: 'Identity' }] }),
}));

global.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/1234');
global.URL.revokeObjectURL = vi.fn();

describe('demo-json-extract', () => {
  it('should extract json', async () => {
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

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const testFile = new File(['123'], 'model.onnx');
    Object.defineProperty(fileInput, 'files', { value: [testFile] });

    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 50));

    expect(document.getElementById('status-text')?.textContent).toBe('Done!');
    expect(document.getElementById('stats-text')?.innerHTML).toContain('model.onnx');

    document.getElementById('download-btn')?.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
