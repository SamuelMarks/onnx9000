import { describe, it, expect, vi } from 'vitest';
import { initProgressiveLoadingDemo } from '../src/main.js';

vi.mock('@onnx9000/backend-web', () => ({
  loadProgressive: vi.fn().mockResolvedValue({
    run: vi.fn().mockResolvedValue({ output: 42 }),
  }),
}));

describe('demo-progressive-loading', () => {
  it('should run progressive loading', async () => {
    document.body.innerHTML = `
      <input id="modelUrl" value="test.onnx" />
      <button id="loadBtn"></button>
      <button id="runBtn" disabled></button>
      <div id="output"></div>
    `;
    initProgressiveLoadingDemo();

    document.getElementById('loadBtn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('output')?.textContent).toContain('Session initialized');

    document.getElementById('runBtn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('output')?.textContent).toContain('Success!');
  });
});
