import { describe, it, expect } from 'vitest';
import { Agent } from '../src/index.js';

describe('Agent', () => {
  it('should run', () => {
    const a = new Agent();
    expect(a.run()).toBe('[agent] processed');
  });
});
