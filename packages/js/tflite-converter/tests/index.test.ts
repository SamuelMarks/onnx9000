import { describe, it, expect } from 'vitest';
import { Tfliteconverter } from '../src/index';
describe('tflite-converter', () => {
  it('runs', () => {
    expect(new Tfliteconverter().run()).toBe('[tflite-converter] processed');
  });
});
