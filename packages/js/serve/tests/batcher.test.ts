import { describe, it, expect, vi } from 'vitest';
import { DynamicBatcher } from '../src/batcher.js';

describe('DynamicBatcher', () => {
  it('should batch requests', async () => {
    vi.useFakeTimers();
    const execute = vi.fn().mockResolvedValue([{ out: 1 }, { out: 2 }]);
    const batcher = new DynamicBatcher(execute, { maxBatchSize: 2 });

    const p1 = batcher.add({ input_ids: [1] });
    const p2 = batcher.add({ input_ids: [1, 2] });

    vi.runAllTimers();
    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1.out).toBe(1);
    expect(r2.out).toBe(2);
    expect(execute).toHaveBeenCalled();
    vi.useRealTimers();
  });
});
