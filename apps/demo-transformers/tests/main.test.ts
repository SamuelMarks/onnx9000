import { describe, it, expect, vi } from 'vitest';
import { initTransformersDemo } from '../src/main.js';

vi.mock('@onnx9000/transformers', () => ({
  pipeline: vi
    .fn()
    .mockResolvedValue(vi.fn().mockResolvedValue({ label: 'positive', score: 0.99 })),
}));

describe('demo-transformers', () => {
  it('should run pipeline', async () => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="transformers-output"></div>';
    initTransformersDemo();
    document.getElementById('run-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('transformers-output')?.textContent).toContain('Success!');
  });
});
