import { describe, it, expect } from 'vitest';
import { ORTTraining } from '../src/index';

describe('ORTTraining', () => {
  it('should process correctly', () => {
    const obj = new ORTTraining();
    expect(obj.process('test')).toBe('ORT Training processed test');
  });
});
