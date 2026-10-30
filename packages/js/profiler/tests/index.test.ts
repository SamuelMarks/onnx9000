import { describe, expect, it } from 'vitest';
import { Profiler } from '../src/index.js';

describe('Profiler', () => {
  it('should record events, compute total duration, and track peak memory', async () => {
    const p = new Profiler('model.onnx');
    await p.run();
    expect(p.peakMemory).toBe(42.5);
    p.recordEvent({ nodeName: 'conv1', durationMs: 2.5, memoryMb: 10.0 });
    expect(p.peakMemory).toBe(42.5);
    expect(p.totalDurationMs()).toBe(4.0);
  });
});
