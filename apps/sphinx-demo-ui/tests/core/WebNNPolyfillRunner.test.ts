/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WebNNPolyfillRunner } from '../../src/core/WebNNPolyfillRunner';
import { WorkerManager } from '../../src/core/WorkerManager';

const mockInstance = {
  initWorker: vi.fn().mockResolvedValue('worker-webnn'),
  execute: vi.fn().mockResolvedValue({ output: new Float32Array([2.0]) }),
  terminate: vi.fn()
};

vi.mock('../../src/core/WorkerManager', () => {
  return {
    WorkerManager: {
      getInstance: vi.fn(() => mockInstance)
    }
  };
});

describe('WebNNPolyfillRunner', () => {
  let runner: WebNNPolyfillRunner;

  beforeEach(() => {
    runner = new WebNNPolyfillRunner();
    vi.clearAllMocks();
  });

  it('should run WebNN execution successfully', async () => {
    mockInstance.execute.mockResolvedValueOnce({ output: new Float32Array([2.0]) });
    const inputBinary = new Uint8Array([0, 1, 2]);
    const inputTensors = { input_1: new Float32Array([0.5]) };

    const result = await runner.runInference(inputBinary, inputTensors);

    expect(result).toHaveProperty('output');
    expect(result.output[0]).toBe(2.0);
    expect(mockInstance.initWorker).toHaveBeenCalledWith('/workers/webnn-worker.js');
    expect(mockInstance.execute).toHaveBeenCalled();
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });

  it('should handle WebNN execution errors', async () => {
    mockInstance.execute.mockRejectedValueOnce(new Error('WebNN failed'));

    const inputBinary = new Uint8Array([0]);
    const inputTensors = {};

    await expect(runner.runInference(inputBinary, inputTensors)).rejects.toThrow('WebNN failed');
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });
});
