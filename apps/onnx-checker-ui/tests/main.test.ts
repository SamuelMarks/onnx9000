import { describe, it, expect, vi } from 'vitest';
import { initOnnxCheckerUI } from '../src/main.js';

vi.mock('@onnx9000/core', () => ({
  ValidationContext: class {
    errors = [];
  },
  check_model: vi.fn(),
}));

describe('onnx-checker-ui', () => {
  it('should validate model', async () => {
    document.body.innerHTML = '<div id="dropzone"></div><div id="results"></div>';
    initOnnxCheckerUI();

    const dropzone = document.getElementById('dropzone')!;
    const e = new Event('drop');
    (e as any).dataTransfer = { files: [new File([''], 'model.onnx')] };
    dropzone.dispatchEvent(e);

    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('results')?.textContent).toContain('structurally valid');
  });
});
