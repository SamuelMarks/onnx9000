import { describe, it, expect } from 'vitest';
import { Gguf } from '../src/index';
describe('gguf', () => {
  it('runs', () => {
    expect(new Gguf().run()).toBe('[gguf] processed');
  });
});
