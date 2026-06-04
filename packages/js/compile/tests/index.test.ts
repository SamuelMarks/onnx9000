import { describe, it, expect } from 'vitest';
import { Compile } from '../src/index.js';
describe('compile', () => {
  it('runs', () => {
    expect(new Compile().run()).toBe('[compile] processed');
  });
});
