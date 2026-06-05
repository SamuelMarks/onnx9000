import { describe, it, expect, vi } from 'vitest';
import { initZooDemo } from '../src/main.js';

vi.mock('@onnx9000/core', () => ({
  fetchSafetensorsHeader: vi.fn().mockResolvedValue({
    headerObj: { __metadata__: { format: 'pt' }, tensor1: {} },
    headerSize: 100,
  }),
  loadTensors: async function* () {
    yield { name: 'tensor1', info: { dtype: 'F32', shape: [10, 10] } };
    yield { name: 'tensor2', info: { dtype: 'F32', shape: [5, 5] } };
  },
}));

describe('demo-zoo', () => {
  it('should fetch and stream', async () => {
    document.body.innerHTML = `
      <button id="fetch-btn"></button>
      <button id="stream-btn" disabled></button>
      <div id="zoo-output"></div>
      <div id="progress-bar"></div>
    `;
    initZooDemo();

    document.getElementById('fetch-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('zoo-output')?.textContent).toContain(
      'Successfully fetched metadata!',
    );

    document.getElementById('stream-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('zoo-output')?.textContent).toContain(
      'Progressively loaded weights',
    );
  });
});
