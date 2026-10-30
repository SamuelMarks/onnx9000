// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOnnxToolDemo } from '../src/main.js';

describe('demo-onnx-tool', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
  });

  it('should initialize and run ONNX tool demo', async () => {
    document.body.innerHTML = '';
    initOnnxToolDemo();

    document.body.innerHTML = '<button id="btn-run"></button>';
    initOnnxToolDemo();
    document.getElementById('btn-run')?.click();

    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOnnxToolDemo();

    const btn = document.getElementById('btn-run') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Running...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] ONNX Tool execution complete.');
  });
});
