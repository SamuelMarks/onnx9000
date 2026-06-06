import { describe, expect, it } from 'vitest';
import * as Module from '../../../../src/providers/webgpu/shaders/bonsai';

describe('bonsai.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
