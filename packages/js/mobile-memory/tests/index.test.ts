import { describe, expect, it } from 'vitest';
import { MobileMemory } from '../src/index.js';

describe('MobileMemory', () => {
  it('should process', () => {
    expect(new MobileMemory().process('test')).toContain('test');
  });
});
