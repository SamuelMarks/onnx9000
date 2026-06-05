import { describe, it, expect, vi } from 'vitest';
import { handleWorkerMessage } from '../src/worker.js';

vi.mock('@onnx9000/c-compiler', () => ({
  compileOnnxToC: vi.fn().mockResolvedValue({ header: 'h', source: 's', summary: 'sum' }),
}));

describe('onnx2c-ui worker', () => {
  it('should compile', async () => {
    const postMessage = vi.fn();
    await handleWorkerMessage(
      { data: { buffer: new Uint8Array(), options: {} } } as any,
      postMessage,
    );
    expect(postMessage).toHaveBeenCalledWith({
      header: 'h',
      source: 's',
      summary: 'sum',
      arenaSize: 250000,
    });
  });
});
