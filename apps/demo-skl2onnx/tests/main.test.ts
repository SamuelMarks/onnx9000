import { describe, it, expect, vi } from 'vitest';
import { initSkl2OnnxDemo } from '../src/main.js';

describe('demo-skl2onnx', () => {
  it('should run conversion', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-convert"></button><div id="output"></div>';
    initSkl2OnnxDemo();
    document.getElementById('btn-convert')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Parsing');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      '[OK] SKL2ONNX conversion complete.',
    );
    vi.useRealTimers();
  });
});
