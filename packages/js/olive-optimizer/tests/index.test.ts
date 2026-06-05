import { describe, it, expect } from 'vitest';
import { OliveOptimizer } from '../src/index.js';

describe('OliveOptimizer', () => {
  it('should run', () => {
    expect(new OliveOptimizer().process('test')).toContain('test');
  });
});
