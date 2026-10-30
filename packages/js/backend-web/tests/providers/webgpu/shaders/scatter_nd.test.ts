import { describe, expect, it } from 'vitest';
import * as Module from '../../../../src/providers/webgpu/shaders/scatter_nd';

describe('scatter_nd.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
