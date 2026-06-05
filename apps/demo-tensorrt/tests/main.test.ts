import { describe, it, expect, vi } from 'vitest';
import { initTensorrtDemo } from '../src/main.js';

describe('demo-tensorrt', () => {
  it('should run trt conversion', () => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
    initTensorrtDemo();
    document.getElementById('convert-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('import tensorrt as trt');
  });
});
