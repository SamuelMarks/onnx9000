import { describe, it, expect, vi } from 'vitest';
import { runCli } from '../src/cli';
import { ModelEnsemble } from '../src/ensemble';
import { KVCacheManager } from '../src/kv_cache';
import { globalLogger } from '../src/logger';

// Mock the serveNode so it doesn't spin up a server
vi.mock('../src/node', () => ({
  serveNode: vi.fn(),
}));

describe('Serve Extra Coverage', () => {
  it('cli.ts coverage', () => {
    runCli([
      '--port',
      '9090',
      '--model-repository',
      './test-models',
      '--max-batch-size',
      '32',
      '--log-verbose',
      '--enable-prometheus',
      '--gpu-only',
      '--http2',
    ]);
    expect(globalLogger.level).toBe(1); // DEBUG is usually 0
  });

  it('ensemble.ts coverage', async () => {
    // Normal graph
    const ensemble = new ModelEnsemble({
      name: 'test',
      inputs: ['input1'],
      outputs: { out: 'node3.out' },
      nodes: [
        { id: 'node1', type: 'model', inputs: { x: 'global.input1' }, outputs: ['output0'] },
        { id: 'node2', type: 'tokenizer', inputs: { x: 'node1.output0' }, outputs: ['input_ids'] },
        {
          id: 'node3',
          type: 'logic',
          inputs: { a: 'node2.input_ids' },
          outputs: ['out'],
          logic: async (i) => i.a,
        },
        { id: 'node4', type: 'post_processor', inputs: {}, outputs: ['out'] },
        { id: 'node5', type: 'lora_adapter', inputs: {}, outputs: ['weights'] },
        {
          id: 'node6',
          type: 'condition',
          inputs: {},
          outputs: ['route'],
          condition: () => 'node7',
        },
      ],
    });

    const res = await ensemble.execute({ input1: 10 }, {});
    expect(res.out).toEqual([101, 2023, 102]);
    expect(res.__metadata__.latencies.node1).toBeDefined();

    // Cycle graph
    expect(() => {
      new ModelEnsemble({
        name: 'cycle',
        inputs: [],
        outputs: {},
        nodes: [
          { id: 'A', type: 'model', inputs: { x: 'B.out' }, outputs: ['out'] },
          { id: 'B', type: 'model', inputs: { x: 'A.out' }, outputs: ['out'] },
        ],
      });
    }).toThrow('Infinite routing loop detected');
  });

  it('kv_cache.ts coverage', async () => {
    vi.useFakeTimers();
    const syncAdapter = {
      save: vi.fn(),
      load: vi.fn().mockResolvedValue(new Float32Array([1, 2, 3])),
      delete: vi.fn(),
    };

    const kv = new KVCacheManager(syncAdapter);

    // Load from adapter
    const data1 = await kv.getCache('session-1');
    expect(data1).toEqual(new Float32Array([1, 2, 3]));
    expect(syncAdapter.load).toHaveBeenCalledWith('session-1');

    // Get from memory
    const data2 = await kv.getCache('session-1');
    expect(data2).toEqual(new Float32Array([1, 2, 3]));

    // Set cache
    await kv.setCache('session-2', new Float32Array([4, 5, 6]), 'hash-1');
    expect(syncAdapter.save).toHaveBeenCalledWith('session-2', expect.any(Float32Array));

    // Prompt cache
    expect(kv.getPromptCache('hash-1')).toEqual(new Float32Array([4, 5, 6]));
    expect(kv.getPromptCache('unknown')).toBeNull();

    // Load unknown from adapter
    syncAdapter.load.mockResolvedValueOnce(null);
    expect(await kv.getCache('session-3')).toBeNull();

    // Eviction
    kv.idleTimeoutMs = 100;
    vi.advanceTimersByTime(200000);
    expect(kv.getPromptCache('hash-1')).toBeNull();

    // Flush
    await kv.setCache('session-4', new Float32Array([0]), 'hash-2');
    await kv.flushAll();
    expect(kv.getPromptCache('hash-2')).toBeNull();

    // Ring Buffer
    const rb = kv.createRingBuffer(10);
    expect(rb.length).toBe(10);

    vi.useRealTimers();
  });
});
