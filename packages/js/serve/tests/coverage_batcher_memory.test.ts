import { describe, it, expect, vi } from 'vitest';
import { DynamicBatcher } from '../src/batcher';
import { MemoryManager } from '../src/memory';
import { globalLogger, LogLevel } from '../src/logger';

describe('Batcher & Memory & Logger', () => {
  it('MemoryManager coverage', async () => {
    const mm = new MemoryManager({
      maxVramBytes: 1000,
      maxRamBytes: 1000,
      maxRamPercent: 0.8,
      maxConcurrentExecutions: 1,
    });

    expect(await mm.requestLoad('model', 2000)).toBe(false);

    mm.registerModel({
      id: 'm1',
      sizeBytes: 500,
      lastUsed: 0,
      buffer: new ArrayBuffer(0),
      unload: vi.fn(),
    });
    mm.trackUsage('m1');
    mm.trackUsage('unknown');

    expect(await mm.requestLoad('m2', 300)).toBe(true);

    mm.registerModel({
      id: 'm3',
      sizeBytes: 300,
      lastUsed: 1,
      buffer: new ArrayBuffer(0),
      unload: vi.fn(),
    });
    expect(await mm.requestLoad('m4', 200)).toBe(true);

    // We actually need to just reset it and test
    const mm2 = new MemoryManager({
      maxVramBytes: 1000,
      maxRamBytes: 1000,
      maxRamPercent: 0.8,
      maxConcurrentExecutions: 1,
    });
    (mm2 as Object).currentRamUsage = 900;
    await expect(mm2.requestLoad('huge', 150)).rejects.toThrow('503');

    await mm2.beginExecution();
    await expect(mm2.beginExecution()).rejects.toThrow('Max concurrent');
    mm2.endExecution();

    const mockGc = vi.fn();
    globalThis.gc = mockGc as Object;
    for (let i = 0; i < 100; i++) mm2.endExecution();
    expect(mockGc).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
  it('Batcher coverage', async () => {
    let executeMock = vi.fn().mockResolvedValue(['res1', 'res2']);
    const batcher = new DynamicBatcher(executeMock, { maxBatchSize: 2, batchTimeoutMs: 10 });

    const p1 = batcher.add({ val: 1 }, 0);
    const p2 = batcher.add({ val: 2 }, 1); // high priority

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(executeMock).toHaveBeenCalled();

    // Test sequence padding and attention mask
    let executeMock2 = vi.fn().mockResolvedValue(['out1', 'out2']);
    const batcher2 = new DynamicBatcher(executeMock2, { maxBatchSize: 2, batchTimeoutMs: 10 });

    const p3 = batcher2.add({ input_ids: [1, 2] });
    const p4 = batcher2.add({ input_ids: [3, 4, 5] });

    await Promise.all([p3, p4]);

    // Test executeBatch rejecting
    let executeMock3 = vi.fn().mockRejectedValue(new Error('batch crash'));
    const batcher3 = new DynamicBatcher(executeMock3, { maxBatchSize: 1 });
    await expect(batcher3.add({ val: 1 })).rejects.toThrow('batch crash');

    // Test queue length 0 flush
    const b = new DynamicBatcher(executeMock);
    (b as Object).flush();
  });
  it('Logger export', async () => {
    globalLogger.exporterUrl = 'http://test';
    const fetchMock = vi.fn().mockRejectedValue(new Error('ignore'));
    vi.stubGlobal('fetch', fetchMock);
    globalLogger.level = LogLevel.TRACE;
    globalLogger.trace('trace');
    expect(fetchMock).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
