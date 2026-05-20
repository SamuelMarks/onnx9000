import { describe, it, expect } from 'vitest';
import { MobileMemory } from '../src/index';

describe('MobileMemory', () => {
  it('should process correctly', () => {
    const obj = new MobileMemory();
    expect(obj.process('test')).toBe('Mobile Memory processed test');
  });
});
