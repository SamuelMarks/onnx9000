import { describe, it, expect } from 'vitest';
import { Graphsurgeon } from '../src/index.js';
describe('graphsurgeon', () => {
  it('runs', () => {
    expect(new Graphsurgeon().run()).toBe('[graphsurgeon] processed');
  });
});
