import { describe, it, expect } from 'vitest';
import { MemoryArena } from '../src/index';

describe('MemoryArena', () => {
  it('should plan layout', () => {
    const arena = new MemoryArena();
    expect(arena.plan('model')).toBe('[Arena] planner processed model');
    expect(() => arena.plan('')).toThrow('Invalid model string');
  });
});
