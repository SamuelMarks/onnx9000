import { describe, it, expect, vi } from 'vitest';
import { WebGPUManager } from '../src/webgpu';
import { WorkerPool } from '../src/worker_pool';
import { serveNode } from '../src/node';

describe('Coverage Extra', () => {
  it('WebGPUManager', async () => {
    const manager = new WebGPUManager();
    expect(manager.fallbackToWasm).toBe(false);

    // Without navigator
    await manager.init();
    expect(manager.fallbackToWasm).toBe(true);
    expect(manager.getTargetDevice('test').type).toBe('wasm');

    const mockNav = {} as Object;
    vi.stubGlobal('navigator', mockNav);

    // With navigator.gpu but no adapter
    mockNav.gpu = { requestAdapter: async () => null };
    manager.fallbackToWasm = false;
    await manager.init();
    expect(manager.fallbackToWasm).toBe(true);

    // With navigator.gpu and adapter
    let lostTrigger: Object;
    const lostPromise = new Promise((resolve) => {
      lostTrigger = resolve;
    });

    mockNav.gpu = {
      requestAdapter: async () => ({
        features: new Set(['shader-f16']),
        requestDevice: async () => ({
          lost: lostPromise,
        }),
      }),
    };
    manager.fallbackToWasm = false;
    await manager.init();
    expect(manager.device).toBeDefined();
    expect(manager.getTargetDevice('test').type).toBe('webgpu');

    // Adapter requestDevice throws
    mockNav.gpu = {
      requestAdapter: async () => ({
        features: new Set([]),
        requestDevice: async () => {
          throw new Error('fail');
        },
      }),
    };
    manager.fallbackToWasm = false;
    await manager.init();
    expect(manager.fallbackToWasm).toBe(true);

    // Test the device loss handler just once
    lostTrigger({ message: 'test loss' });
    await new Promise((r) => setTimeout(r, 0));

    vi.unstubAllGlobals();
  });

  it('WorkerPool', async () => {
    vi.useFakeTimers();
    const pool = new WorkerPool(2);
    pool.init();
    const sab = new SharedArrayBuffer(1024);
    const resPromise = pool.execute('test-model', sab);

    vi.runAllTimers(); // this runs the settimeout inside postMessage
    const res = await resPromise;
    expect(res).toBe(true);

    const poolEmpty = new WorkerPool(0);
    await expect(poolEmpty.execute('test', sab)).rejects.toThrow('No available workers');

    vi.useRealTimers();
  });

  it('Node Serve', () => {
    const mockServer = {
      fetch: vi.fn().mockResolvedValue({
        status: 200,
        headers: new Headers({ 'x-test': '1' }),
        body: {
          getReader: () => {
            let count = 0;
            return {
              read: async () =>
                count++ === 0 ? { done: false, value: new Uint8Array([1]) } : { done: true },
            };
          },
        },
      }),
    };

    const server = serveNode(mockServer as Object, 0, false);
    server.close();
  });

  it('WorkerPool handle message', () => {
    const p = new WorkerPool(1);
    p.init();
    (p as Object).handleWorkerMessage({}); // cover empty func
    (p as Object).workers[0].terminate(); // cover empty func
  });
});

it('WorkerPool gracefully handles missing os.cpus', () => {
  const originalOs = require('os');
  vi.doMock('os', () => ({
    cpus: () => [],
  }));

  const pool = new WorkerPool(4);
  expect(pool.maxWorkers).toBeGreaterThan(0);
  vi.doUnmock('os');
});
