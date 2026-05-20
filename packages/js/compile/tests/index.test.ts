import { describe, it, expect } from 'vitest';
import { Compile } from '../src/index';
describe('compile', () => {
  it('runs', () => {
    expect(new Compile().run()).toBe('[compile] processed');
  });
});
