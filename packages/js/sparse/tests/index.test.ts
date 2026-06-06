import { describe, expect, it } from 'vitest';
import { Sparse } from '../src/index.js';

describe('Sparse', () => {
  it('should run', () => {
    expect(new Sparse().run()).toBeDefined();
  });
});
