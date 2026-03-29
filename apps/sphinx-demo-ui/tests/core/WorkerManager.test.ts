/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WorkerManager, WorkerMessage } from '../../src/core/WorkerManager';

// Mock global Worker
class MockWorker {
  onmessage: object = null;
  onerror: object = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor(_url: string, _opts: object) {}
}

global.Worker = MockWorker as object;

describe('WorkerManager', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    WorkerManager.getInstance().terminate();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should be a singleton', () => {
    const wm1 = WorkerManager.getInstance();
    const wm2 = WorkerManager.getInstance();
    expect(wm1).toBe(wm2);
  });

  it('should initialize and terminate a worker safely', () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    expect((wm as object).worker).toBeDefined();

    wm.terminate();
    expect((wm as object).worker).toBeNull();
  });

  it('should reject execute if worker not initialized', async () => {
    const wm = WorkerManager.getInstance();
    await expect(wm.execute('test', {})).rejects.toThrow('Worker not initialized.');
  });

  it('should execute and resolve upon matching message', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const worker = (wm as object).worker as MockWorker;

    // Catch the ID when postMessage is called
    let capturedId = '';
    worker.postMessage.mockImplementation((msg: WorkerMessage) => {
      capturedId = msg.id;
    });

    const promise = wm.execute('test', { data: 123 });

    // Simulate worker returning success
    worker.onmessage({
      data: { id: capturedId, type: 'test_RESPONSE', payload: { success: true } }
    });

    const result = await promise;
    expect(result).toEqual({ success: true });
  });

  it('should reject upon matching error message', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const worker = (wm as object).worker as MockWorker;

    let capturedId = '';
    worker.postMessage.mockImplementation((msg: WorkerMessage) => {
      capturedId = msg.id;
    });

    const promise = wm.execute('test', { data: 123 });

    worker.onmessage({
      data: { id: capturedId, type: 'test_RESPONSE', error: 'Internal Worker Error' }
    });

    await expect(promise).rejects.toThrow('Internal Worker Error');
  });

  it('should reject upon worker onerror crash', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const worker = (wm as object).worker as MockWorker;
    const promise = wm.execute('test', {});

    // Simulate crash
    worker.onerror(new Error('Fatal Error'));

    await expect(promise).rejects.toThrow('Worker crashed');
  });

  it('should reject pending requests on terminate', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const promise = wm.execute('test', {});
    wm.terminate();

    await expect(promise).rejects.toThrow('Worker forcefully terminated');
  });

  it('should timeout and auto-terminate runaway worker', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const promise = wm.execute('test', {}, 100); // 100ms timeout

    // Fast forward past timeout
    vi.advanceTimersByTime(150);

    await expect(promise).rejects.toThrow('Worker task timed out after 100ms');

    // Manager should have restarted the worker
    expect((wm as object).worker).not.toBeNull();
  });

  it('should handle unmapped streaming stdout gracefully without resolving pending requests', async () => {
    const wm = WorkerManager.getInstance();
    wm.initWorker();

    const worker = (wm as object).worker as MockWorker;

    // Stream a log
    worker.onmessage({
      data: { id: 'unknown-id', type: 'STREAM_STDOUT', payload: 'Compiling...' }
    });

    // Stream is handled async via EventBus in another test, but here we just ensure
    // it doesn't crash the worker or resolve a pending request erroneously.
    expect((wm as object).pendingRequests.size).toBe(0);
  });
});

it('should terminate existing worker before initializing a new one', () => {
  const wm = WorkerManager.getInstance();
  wm.initWorker();

  const terminateSpy = vi.spyOn(wm, 'terminate');

  // Calling it again should trigger termination of the previous one
  wm.initWorker();

  expect(terminateSpy).toHaveBeenCalledTimes(1);

  terminateSpy.mockRestore();
});

it('should handle fallback missing randomUUID', () => {
  // Save orig
  const origCrypto = global.crypto;
  Object.defineProperty(global, 'crypto', {
    value: { ...origCrypto, randomUUID: undefined },
    writable: true,
    configurable: true
  });

  const wm = WorkerManager.getInstance();
  wm.initWorker();

  const worker = (wm as object).worker as MockWorker;
  let capturedId = '';
  worker.postMessage.mockImplementation((msg: WorkerMessage) => {
    capturedId = msg.id;
  });

  wm.execute('test', {});
  expect(capturedId).toBeDefined();
  expect(capturedId.length).toBeGreaterThan(5);

  // Restore
  Object.defineProperty(global, 'crypto', {
    value: origCrypto,
    writable: true,
    configurable: true
  });
});
