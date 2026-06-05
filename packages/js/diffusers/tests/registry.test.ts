import { describe, it, expect } from 'vitest';
import { register_op } from '../src/registry.js';

describe('diffusers registry', () => {
  it('should register', () => {
    @register_op('domain', 'op')
    class Mock {}
    expect((Mock as any).opName).toBe('op');
  });
});
