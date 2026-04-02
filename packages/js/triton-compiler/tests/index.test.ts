import { describe, it, expect } from 'vitest';
import * as triton from '../src/index.js';

describe('Triton Compiler Index', () => {
  it('should export compiler functions', () => {
    expect(triton).toBeDefined();
  });
});
