import { describe, it, expect } from 'vitest';
import { Sparse } from '../src/index.js';
describe('sparse', () => {
  it('runs', () => {
    expect(new Sparse().run()).toBe('[sparse] processed');
  });
});
