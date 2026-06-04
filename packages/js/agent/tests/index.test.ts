import { describe, it, expect } from 'vitest';
import { Agent } from '../src/index.js';
describe('agent', () => {
  it('runs', () => {
    expect(new Agent().run()).toBe('[agent] processed');
  });
});
