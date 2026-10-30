import { describe, expect, it, vi } from 'vitest';
import { handleWorkerMessage } from '../src/worker';

type WorkerEvent = Parameters<typeof handleWorkerMessage>[0];

vi.mock('@onnx9000/c-compiler', () => ({
  compileOnnxToC: vi.fn().mockImplementation((buffer, _options) => {
    if (buffer === 'error') return Promise.reject(new Error('Compile failed'));
    if (buffer === 'str_error') return Promise.reject('String compile failed');
    return Promise.resolve({
      header: 'mock header',
      source: 'mock source',
      summary: 'mock summary',
    });
  }),
}));

describe('handleWorkerMessage', () => {
  it('should process valid request successfully', async () => {
    const postMessage = vi.fn();
    const event = {
      data: { buffer: new Uint8Array([1, 2, 3]), options: {} },
    } as WorkerEvent;

    await handleWorkerMessage(event, postMessage);

    expect(postMessage).toHaveBeenCalledWith({
      header: 'mock header',
      source: 'mock source',
      summary: 'mock summary',
      arenaSize: 250000,
    });
  });

  it('should handle compilation errors', async () => {
    const postMessage = vi.fn();
    const event = {
      data: { buffer: 'error', options: {} },
    } as WorkerEvent;

    await handleWorkerMessage(event, postMessage);

    expect(postMessage).toHaveBeenCalledWith({
      error: 'Compile failed',
    });

    await handleWorkerMessage(
      { data: { buffer: 'str_error', options: {} } } as WorkerEvent,
      postMessage,
    );
    expect(postMessage).toHaveBeenCalledWith({
      error: 'String compile failed',
    });

    if (typeof self !== 'undefined' && self.onmessage) {
      const spy = vi.spyOn(self, 'postMessage').mockImplementation(() => {});
      await self.onmessage(event);
      spy.mockRestore();
    }
  });
});
