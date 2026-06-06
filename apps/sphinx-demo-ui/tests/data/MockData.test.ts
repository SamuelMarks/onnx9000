import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/data/MockData';

describe('MockData.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
