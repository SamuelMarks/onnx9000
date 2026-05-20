import { describe, it, expect } from 'vitest';
import { Hummingbird } from '../src/index';
describe('hummingbird', () => {
  it('runs', () => {
    expect(new Hummingbird().run()).toBe('[hummingbird] processed');
  });
});
