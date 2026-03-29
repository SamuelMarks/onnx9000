import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OliveOptimizer } from '../../src/core/OliveOptimizer';
import { WorkerManager } from '../../src/core/WorkerManager';

const mockInstance = {
  initWorker: vi.fn().mockResolvedValue('worker-123'),
  execute: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
  terminate: vi.fn()
};

vi.mock('../../src/core/WorkerManager', () => {
  return {
    WorkerManager: {
      getInstance: vi.fn(() => mockInstance)
    }
  };
});

describe('OliveOptimizer', () => {
  let optimizer: OliveOptimizer;

  beforeEach(() => {
    optimizer = new OliveOptimizer();
    vi.clearAllMocks();
  });

  it('should optimize ONNX binary successfully', async () => {
    mockInstance.execute.mockResolvedValueOnce(new Uint8Array([1, 2, 3]));
    const input = new Uint8Array([0, 0, 0, 0]);
    const config = {
      quantizationLevel: 'FP16' as const,
      enableStaticShapeInference: true,
      enableTransformerFusion: false
    };

    const result = await optimizer.optimize(input, config);
    expect(result).toBeInstanceOf(Uint8Array);
    expect(mockInstance.initWorker).toHaveBeenCalledWith('/workers/olive-worker.js');
    expect(mockInstance.execute).toHaveBeenCalled();
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });

  it('should handle optimization errors', async () => {
    mockInstance.execute.mockRejectedValueOnce(new Error('Optimization failed'));

    const input = new Uint8Array([0]);
    const config = {
      quantizationLevel: 'None' as const,
      enableStaticShapeInference: false,
      enableTransformerFusion: false
    };

    await expect(optimizer.optimize(input, config)).rejects.toThrow('Optimization failed');
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });

  it('should simplify ONNX binary successfully', async () => {
    mockInstance.execute.mockResolvedValueOnce(new Uint8Array([1, 2, 3]));
    const input = new Uint8Array([0, 0, 0, 0]);

    const result = await optimizer.simplify(input);
    expect(result).toBeInstanceOf(Uint8Array);
    expect(mockInstance.initWorker).toHaveBeenCalledWith('/workers/simplifier-worker.js');
    expect(mockInstance.execute).toHaveBeenCalled();
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });

  it('should handle simplification errors', async () => {
    mockInstance.execute.mockRejectedValueOnce(new Error('Simplification failed'));

    const input = new Uint8Array([0]);

    await expect(optimizer.simplify(input)).rejects.toThrow('Simplification failed');
    expect(mockInstance.terminate).toHaveBeenCalledWith();
  });
});
