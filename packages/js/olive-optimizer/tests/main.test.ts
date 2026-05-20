import { describe, it, expect } from 'vitest';
import { OliveOptimizer } from '../src/index';

describe('OliveOptimizer', () => {
  it('should process correctly', () => {
    const obj = new OliveOptimizer();
    expect(obj.process('test')).toBe('Olive Optimizer processed test');
  });
});
