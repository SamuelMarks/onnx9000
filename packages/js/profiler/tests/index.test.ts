import { describe, it, expect } from 'vitest';
import { Profiler } from '../src/index.js';

describe('Profiler', () => {
  it('should run', async () => {
    const p = new Profiler('test');
    await p.run();
    expect(p.peakMemory).toBe(42.5);
  });
});
