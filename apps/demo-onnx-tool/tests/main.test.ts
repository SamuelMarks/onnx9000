import { describe, it, expect, vi } from 'vitest';
import { initOnnxToolDemo } from '../src/main.js';

describe('demo-onnx-tool', () => {
  it('should run tool', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOnnxToolDemo();
    document.getElementById('btn-run')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Running');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      '[OK] ONNX Tool execution complete.',
    );
    vi.useRealTimers();
  });
});
