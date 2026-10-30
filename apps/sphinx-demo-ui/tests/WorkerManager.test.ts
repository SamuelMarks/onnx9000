import { describe, expect, it } from 'vitest';
import { WorkerManager } from '../src/core/WorkerManager.js';

describe('WorkerManager', () => {
  it('should init and execute', async () => {
    const mgr = WorkerManager.getInstance();

    class MockWorker {
      onmessage: ((ev: { data: unknown }) => void) | null = null;
      onerror: ((ev: unknown) => void) | null = null;
      postMessage(data: { id: string }) {
        setTimeout(() => {
          if (this.onmessage) {
            this.onmessage({ data: { id: data.id, type: 'RES', payload: { ok: true } } });
          }
        }, 10);
      }
      terminate() {}
    }

    global.Worker = MockWorker as never as typeof Worker;

    mgr.initWorker();
    const res = (await mgr.execute('TEST', {})) as { ok: boolean };
    expect(res.ok).toBe(true);
    mgr.terminate();
  });
});
