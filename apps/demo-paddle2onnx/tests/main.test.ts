import { describe, it, expect, vi } from 'vitest';
import { initPaddle2OnnxDemo } from '../src/main.js';

describe('demo-paddle2onnx', () => {
  it('should run conversion', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
    initPaddle2OnnxDemo();
    document.getElementById('btn-convert')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Parsing');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      '[OK] Paddle2ONNX conversion complete.',
    );
    vi.useRealTimers();
  });
});
