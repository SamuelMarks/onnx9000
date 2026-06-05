// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { WorkerManager } from '../src/core/WorkerManager.js';

describe('WorkerManager', () => {
  it('should init and execute', async () => {
    const mgr = WorkerManager.getInstance();

    class MockWorker {
      onmessage: any;
      onerror: any;
      postMessage(data: any) {
        setTimeout(() => {
          if (this.onmessage) {
            this.onmessage({ data: { id: data.id, type: 'RES', payload: { ok: true } } });
          }
        }, 10);
      }
      terminate() {}
    }

    (global as any).Worker = MockWorker;

    mgr.initWorker();
    const res: any = await mgr.execute('TEST', {});
    expect(res.ok).toBe(true);
    mgr.terminate();
  });
});
