import { describe, expect, it } from 'vitest';
import { MemoryArena } from '../src/index.js';

describe('MemoryArena', () => {
  it('should plan layout from model string and validate input', () => {
    const arena = new MemoryArena();
    expect(arena.plan('model')).toBe('[Arena] planner processed model');
    expect(arena.peakBytes).toBe(32);
    expect(() => arena.plan('')).toThrow('Invalid model string');
    expect(() => new MemoryArena(0)).toThrow('Alignment must be positive');
  });

  it('should allocate non-overlapping tensor lifetimes with slab reuse', () => {
    const arena = new MemoryArena(16);
    expect(arena.alignSize(16)).toBe(16);
    expect(arena.alignSize(20)).toBe(32);

    const blocks = arena.allocateSpecs([
      { name: 't0', byteLength: 24, firstStep: 0, lastStep: 1 },
      { name: 't1', byteLength: 16, firstStep: 1, lastStep: 2 },
      { name: 't2', byteLength: 32, firstStep: 2, lastStep: 3 },
    ]);

    expect(blocks).toHaveLength(3);
    expect(blocks[0]?.offset).toBe(0);
    expect(blocks[1]?.offset).toBe(32);
    // t0 is freed before step 2 and reused for t2
    expect(blocks[2]?.offset).toBe(0);
    expect(arena.peakBytes).toBe(48);

    const view = arena.getView('t2');
    expect(view.byteLength).toBe(32);
    expect(() => arena.getView('missing')).toThrow(
      "Tensor 'missing' is not allocated in the arena",
    );
  });
});
