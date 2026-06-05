import { describe, it, expect, vi } from 'vitest';
import { initDiffusersDemo } from '../src/main.js';

vi.mock('@onnx9000/diffusers/src/pipeline.js', () => ({
  DiffusionPipeline: class {
    constructor() {}
  },
}));

describe('demo-diffusers', () => {
  it('should initialize pipeline', async () => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initDiffusersDemo();
    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    btn.click();
    await new Promise((r) => setTimeout(r, 10)); // flush promises
    expect(document.getElementById('output')?.innerText).toContain('Pipeline initialized');
  });
});
