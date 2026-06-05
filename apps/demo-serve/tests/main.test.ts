import { describe, it, expect, vi } from 'vitest';
import { initServeDemo } from '../src/main.js';

vi.mock('@onnx9000/serve', () => ({
  createServer: vi.fn().mockReturnValue({
    fetch: vi.fn().mockResolvedValue({
      status: 200,
      text: vi.fn().mockResolvedValue('ok'),
    }),
  }),
}));

describe('demo-serve', () => {
  it('should run serverless inference', async () => {
    document.body.innerHTML = `
      <button id="start-btn"></button>
      <button id="req-btn" disabled></button>
      <div id="server-output"></div>
    `;
    initServeDemo();

    document.getElementById('start-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('server-output')?.textContent).toContain('Server initialized');

    document.getElementById('req-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('server-output')?.textContent).toContain(
      'Success! Edge routing',
    );
  });
});
