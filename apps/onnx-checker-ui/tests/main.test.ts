import * as core from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initOnnxCheckerUI } from '../src/main';

vi.mock('@onnx9000/core', () => {
  return {
    ValidationContext: class {
      errors: string[] = [];
    },
    check_model: vi.fn((model, ctx) => {
      if (model.ir_version === 0) ctx.errors.push('Invalid IR version');
    }),
  };
});

describe('initOnnxCheckerUI', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="dropzone"></div><div id="results"></div>';
    vi.clearAllMocks();
  });

  it('should initialize and attach event listeners', () => {
    initOnnxCheckerUI();
    const dropzone = document.getElementById('dropzone')!;

    // Drag events
    dropzone.dispatchEvent(new Event('dragover'));
    expect(dropzone.style.background).toBe('rgb(224, 255, 224)'); // #e0ffe0

    dropzone.dispatchEvent(new Event('dragleave'));
    expect(dropzone.style.background).toBe('rgb(255, 255, 255)'); // #fff
  });

  it('should handle missing elements safely', () => {
    document.body.innerHTML = ''; // missing elements
    expect(() => initOnnxCheckerUI()).not.toThrow();
  });

  it('should process dropped file and validate (success)', async () => {
    initOnnxCheckerUI();
    const dropzone = document.getElementById('dropzone')!;

    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };

    dropzone.dispatchEvent(dropEvent);

    // Wait for async processing
    await new Promise((r) => setTimeout(r, 0));

    const results = document.getElementById('results')!;
    expect(results.innerHTML).toContain('Model test.onnx is structurally valid!');
  });

  it('should skip drop if no files', async () => {
    initOnnxCheckerUI();
    const dropzone = document.getElementById('dropzone')!;

    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [] };

    dropzone.dispatchEvent(dropEvent);
    expect(document.getElementById('results')?.innerHTML).toBe('');
  });

  it('should process dropped file and validate (errors)', async () => {
    vi.mocked(core.check_model).mockImplementationOnce((_model, ctx: any) => {
      ctx.errors.push('Mocked validation error');
    });

    initOnnxCheckerUI();
    const dropzone = document.getElementById('dropzone')!;

    const mockFile = new File(['mock'], 'test.onnx', {
      type: 'application/octet-stream',
    });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };

    dropzone.dispatchEvent(dropEvent);

    // Wait for async processing
    await new Promise((r) => setTimeout(r, 0));

    const results = document.getElementById('results')!;
    expect(results.innerHTML).toContain('Validation Failed');
    expect(results.innerHTML).toContain('Mocked validation error');
  });

  it('should handle exceptions during processing', async () => {
    initOnnxCheckerUI();
    const dropzone = document.getElementById('dropzone')!;

    // file with arrayBuffer that throws
    const mockFile = {
      name: 'test.onnx',
      arrayBuffer: () => Promise.reject(new Error('Read failed')),
    };

    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };

    dropzone.dispatchEvent(dropEvent);

    await new Promise((r) => setTimeout(r, 0));

    const results = document.getElementById('results')!;
    expect(results.innerHTML).toContain('Error: Read failed');
  });
});
