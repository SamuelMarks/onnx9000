import { describe, it, expect } from 'vitest';
import { MemoryArena } from '../src/index.js';

describe('MemoryArena', () => {
  it('should plan', () => {
    const arena = new MemoryArena();
    expect(arena.plan('test')).toContain('[Arena] planner processed test');
    expect(() => arena.plan('')).toThrow('Invalid model string');
  });
});
