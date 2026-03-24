import { describe, it, expect, vi } from 'vitest';
import { WorkerManager } from '../../src/core/WorkerManager';

class MockWorker {
  onmessage: any = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor() {}
}

global.Worker = MockWorker as any;

describe('WorkerManager Streaming', () => {
  it('should cover the unmapped branch when msg.type is not STREAM_STDOUT', () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();
    const worker = (wm as any).worker as MockWorker;

    // A message with no pending request and not STREAM_STDOUT
    expect(() => {
      worker.onmessage({
        data: { id: 'unknown', type: 'SOME_RANDOM_MSG' }
      });
    }).not.toThrow();
  });
});
