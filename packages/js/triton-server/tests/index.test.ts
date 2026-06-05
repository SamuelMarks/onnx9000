import { describe, it, expect } from 'vitest';
import { TritonServer } from '../src/index.js';

describe('TritonServer', () => {
  it('should run', () => {
    expect(new TritonServer().process('test')).toContain('Triton Server');
  });
});
