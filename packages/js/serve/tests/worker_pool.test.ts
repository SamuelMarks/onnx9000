import { describe, it, expect } from 'vitest';
import { WorkerPool } from '../src/worker_pool.js';

describe('WorkerPool', () => {
  it('should exec', async () => {
    const pool = new WorkerPool();
    pool.init();
    const res = await pool.execute('test', new ArrayBuffer(10) as any);
    expect(res).toBe(true);
  });
});
