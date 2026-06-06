// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { LHS_EXAMPLES, LHS_FRAMEWORKS, RHS_TARGETS } from '../src/data/MockData.js';

describe('MockData', () => {
  it('should be valid', () => {
    expect(LHS_FRAMEWORKS.length).toBeGreaterThan(0);
    expect(Object.keys(LHS_EXAMPLES).length).toBeGreaterThan(0);
    expect(Object.keys(RHS_TARGETS).length).toBeGreaterThan(0);
  });
});
