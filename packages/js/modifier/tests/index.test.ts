import { describe, it, expect } from 'vitest';
import * as modifier from '../src/index.js';

describe('Index exports', () => {
  it('exports GraphMutator and GraphValidator', () => {
    expect(modifier.GraphMutator).toBeDefined();
    expect(modifier.GraphValidator).toBeDefined();
  });
});
