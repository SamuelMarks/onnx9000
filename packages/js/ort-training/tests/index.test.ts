import { describe, expect, it } from 'vitest';
import { ORTTraining } from '../src/index.js';

describe('ORTTraining', () => {
  it('should run', () => {
    expect(new ORTTraining().process('test')).toContain('ORT Training');
  });
});
