import { describe, it, expect } from 'vitest';
import { Mmdnn } from '../src/index';
describe('mmdnn', () => {
  it('runs', () => {
    expect(new Mmdnn().run()).toBe('[mmdnn] processed');
  });
});
