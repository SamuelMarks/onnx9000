import { describe, it, expect, vi } from 'vitest';
import { initTfjsShimDemo } from '../src/main.js';

vi.mock('@onnx9000/tfjs-shim', () => ({
  tidy: (fn: any) => fn(),
  tensor2d: vi.fn().mockReturnValue({ shape: [2, 2], dataSync: () => [1, 2, 3, 4] }),
  matMul: vi.fn().mockReturnValue({ shape: [2, 2], dataSync: () => [19, 22, 43, 50] }),
  relu: vi.fn().mockReturnValue({ shape: [2, 2], dataSync: () => [0, 0, 1, 2] }),
  sub: vi.fn().mockReturnValue({ shape: [2, 2], dataSync: () => [-1, 0, 1, 2] }),
  scalar: vi.fn().mockReturnValue({ shape: [], dataSync: () => [2] }),
}));

describe('demo-tfjs-shim', () => {
  it('should run operations', () => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initTfjsShimDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain(
      'Operations completed inside tf.tidy scope.',
    );
  });
});
