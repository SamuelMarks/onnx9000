// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { WebNNPolyfillRunner } from '../src/core/WebNNPolyfillRunner.js';

vi.mock('../src/core/WorkerManager.js', () => ({
  WorkerManager: {
    getInstance: vi.fn().mockReturnValue({
      initWorker: vi.fn(),
      execute: vi.fn().mockResolvedValue({ y: {} }),
      terminate: vi.fn()
    })
  }
}));

describe('WebNNPolyfillRunner', () => {
  it('should run inference', async () => {
    const runner = new WebNNPolyfillRunner();
    const res = await runner.runInference(new Uint8Array(), {});
    expect(res).toBeDefined();
    expect(res.y).toBeDefined();
  });
});
