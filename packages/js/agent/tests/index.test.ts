import { describe, it, expect } from 'vitest';
import { Agent } from '../src/index';

describe('Agent', () => {
  it('should instantiate and run', () => {
    const agent = new Agent();
    expect(agent.run()).toBe('[agent] processed');
  });
});
