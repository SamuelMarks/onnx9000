import { describe, it, expect, vi } from 'vitest';
import { initTritonDemo } from '../src/main.js';

vi.mock('@onnx9000/triton-compiler', () => ({
  generateTriton: vi.fn().mockReturnValue('def custom_fused_kernel(A, B, C):'),
}));

describe('demo-triton', () => {
  it('should generate triton code', () => {
    document.body.innerHTML = '<button id="generate-btn"></button><div id="output"></div>';
    initTritonDemo();
    document.getElementById('generate-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('def custom_fused_kernel');
  });
});
