import { describe, it, expect } from 'vitest';
import { MemoryArena } from '../src/index';

describe('MemoryArena', () => {
  it('should plan memory', () => {
    const arena = new MemoryArena();
    expect(arena.plan('model_data')).toBe('[Arena] planner processed model_data');
  });

  it('should throw on empty string', () => {
    const arena = new MemoryArena();
    expect(() => arena.plan('')).toThrow('Invalid model string');
  });
});
