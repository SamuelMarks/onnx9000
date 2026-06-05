import { describe, it, expect } from 'vitest';
import { translateNhwcToNchw, calculatePaddingSame } from '../../src/keras/layout.js';

describe('layout', () => {
  it('should translate nhwc', () => {
    expect(translateNhwcToNchw([1, 224, 224, 3])).toEqual([1, 3, 224, 224]);
  });

  it('should calc padding same', () => {
    const pad = calculatePaddingSame(10, 3, 1);
    expect(pad).toEqual([1, 1]);
  });
});
