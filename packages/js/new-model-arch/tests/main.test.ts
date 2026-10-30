import { describe, expect, it } from 'vitest';
import { NewModelArch } from '../src/index.js';

describe('NewModelArch', () => {
  it('should process correctly', () => {
    const obj = new NewModelArch();
    expect(obj.process('test')).toBe('New Model Arch processed test');
  });
});
