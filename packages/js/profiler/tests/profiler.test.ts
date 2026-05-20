import { describe, it, expect } from 'vitest';
import { Profiler } from '../src/index';

describe('Profiler', () => {
  it('should calculate peak memory', async () => {
    const profiler = new Profiler('test.onnx');
    expect(profiler.peakMemory).toBe(0);
    await profiler.run();
    expect(profiler.peakMemory).toBe(42.5);
  });
});
