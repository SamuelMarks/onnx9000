// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { OliveOptimizer } from '../src/core/OliveOptimizer.js';

vi.mock('../src/core/WorkerManager.js', () => ({
  WorkerManager: {
    getInstance: vi.fn().mockReturnValue({
      initWorker: vi.fn(),
      execute: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
      terminate: vi.fn(),
    }),
  },
}));

describe('OliveOptimizer', () => {
  it('should optimize', async () => {
    const opt = new OliveOptimizer();
    const res = await opt.optimize(new Uint8Array(), {
      quantizationLevel: 'None',
      enableStaticShapeInference: false,
      enableTransformerFusion: false,
    });
    expect(res.length).toBe(3);
  });

  it('should simplify', async () => {
    const opt = new OliveOptimizer();
    const res = await opt.simplify(new Uint8Array());
    expect(res.length).toBe(3);
  });
});
