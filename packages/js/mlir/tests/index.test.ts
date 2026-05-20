import { describe, it, expect } from 'vitest';
import { Mlir } from '../src/index';
describe('mlir', () => {
  it('runs', () => {
    expect(new Mlir().run()).toBe('[mlir] processed');
  });
});
