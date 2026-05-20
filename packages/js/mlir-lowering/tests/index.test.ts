import { describe, it, expect } from 'vitest';
import { Mlirlowering } from '../src/index';
describe('mlir-lowering', () => {
  it('runs', () => {
    expect(new Mlirlowering().run()).toBe('[mlir-lowering] processed');
  });
});
