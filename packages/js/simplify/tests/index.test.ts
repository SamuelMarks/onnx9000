import { describe, it, expect } from 'vitest';
import { Simplify } from '../src/index';
describe('simplify', () => {
  it('runs', () => {
    expect(new Simplify().run()).toBe('[simplify] processed');
  });
});
