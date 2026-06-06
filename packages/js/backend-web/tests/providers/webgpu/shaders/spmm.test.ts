import { describe, expect, it } from 'vitest';
import * as Module from '../../../../src/providers/webgpu/shaders/spmm';

describe('spmm.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
