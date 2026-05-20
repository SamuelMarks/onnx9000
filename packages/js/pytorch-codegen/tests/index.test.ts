import { describe, it, expect } from 'vitest';
import { Pytorchcodegen } from '../src/index';
describe('pytorch-codegen', () => {
  it('runs', () => {
    expect(new Pytorchcodegen().run()).toBe('[pytorch-codegen] processed');
  });
});
