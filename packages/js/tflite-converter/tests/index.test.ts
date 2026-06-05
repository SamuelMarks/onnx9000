import { describe, it, expect } from 'vitest';
import { Tfliteconverter } from '../src/index.js';

describe('Tfliteconverter', () => {
  it('should run', () => {
    expect(new Tfliteconverter().run()).toBeDefined();
  });
});
