/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ORTWebRunner } from '../../src/core/ORTWebRunner';
import { WorkerManager } from '../../src/core/WorkerManager';

const mockInstance = {
  initWorker: vi.fn().mockResolvedValue('worker-ort'),
  execute: vi.fn().mockResolvedValue({ output: new Float32Array([1.0]) }),
  terminate: vi.fn()
};

vi.mock('../../src/core/WorkerManager', () => {
  return {
    WorkerManager: {
      getInstance: vi.fn(() => mockInstance)
    }
  };
});

describe('ORTWebRunner', () => {
  let runner: ORTWebRunner;

  beforeEach(() => {
    runner = new ORTWebRunner();
    vi.clearAllMocks();
  });

  it('should run ORT execution successfully', async () => {
    mockInstance.execute.mockResolvedValueOnce({ output: new Float32Array([1.0]) });
    const inputBinary = new Uint8Array([0, 1, 2]);
    const inputTensors = { input_1: new Float32Array([0.5]) };

    const result = await runner.runInference(inputBinary, inputTensors, 'webgl');

    expect(result).toHaveProperty('output');
    expect(result.output[0]).toBe(1.0);
    expect(mockInstance.initWorker).toHaveBeenCalledWith('/workers/ort-worker.js');
    expect(mockInstance.execute).toHaveBeenCalled();
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });

  it('should handle ORT execution errors', async () => {
    mockInstance.execute.mockRejectedValueOnce(new Error('ORT failed'));

    const inputBinary = new Uint8Array([0]);
    const inputTensors = {};

    await expect(runner.runInference(inputBinary, inputTensors, 'wasm')).rejects.toThrow(
      'ORT failed'
    );
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });
});
