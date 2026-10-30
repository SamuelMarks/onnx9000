import { describe, expect, it, vi } from 'vitest';
import { WorkerManager } from '../../src/core/WorkerManager';

class MockWorker {
  onmessage: ((e: { data: { id: string; type: string } }) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
}

global.Worker = MockWorker as never as typeof Worker;

describe('WorkerManager Streaming', () => {
  it('should cover the unmapped branch when msg.type is not STREAM_STDOUT', () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();
    const worker = (wm as never as { worker: MockWorker }).worker;

    // A message with no pending request and not STREAM_STDOUT
    expect(() => {
      worker.onmessage?.({
        data: { id: 'unknown', type: 'SOME_RANDOM_MSG' },
      });
    }).not.toThrow();
  });
});
