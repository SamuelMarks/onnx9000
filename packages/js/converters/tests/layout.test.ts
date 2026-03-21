import { describe, it, expect } from 'vitest';
import {
  translateNhwcToNchw,
  transposeConv2DWeights,
  transposeConv1DWeights,
  transposeConv3DWeights,
  transposeDenseWeights,
  calculatePaddingSame,
  calculatePaddingValid,
} from '../src/keras/layout.js';

describe('layout transpositions', () => {
  it('translateNhwcToNchw works for 3D, 4D, 5D', () => {
    expect(translateNhwcToNchw([1, 10, 3])).toEqual([1, 3, 10]);
    expect(translateNhwcToNchw([1, 28, 28, 3])).toEqual([1, 3, 28, 28]);
    expect(translateNhwcToNchw([1, 16, 28, 28, 3])).toEqual([1, 3, 16, 28, 28]);
    expect(translateNhwcToNchw([1, 2])).toEqual([1, 2]); // unmodified
  });

  it('transposeConv2DWeights', () => {
    // [H, W, I, O] -> [O, I, H, W]
    // 1x1x1x2
    const input = new Float32Array([1.0, 2.0]);
    const out = transposeConv2DWeights(input, 1, 1, 1, 2);
    expect(Array.from(out)).toEqual([1.0, 2.0]); // For 1x1x1x2 it just reads sequentially out

    // 2x1x1x1
    const input2 = new Float32Array([1.0, 2.0]); // H=2
    const out2 = transposeConv2DWeights(input2, 2, 1, 1, 1);
    expect(Array.from(out2)).toEqual([1.0, 2.0]);
  });

  it('transposeConv1DWeights', () => {
    const input = new Float32Array([1.0, 2.0, 3.0, 4.0]); // L=2, I=1, O=2
    const out = transposeConv1DWeights(input, 2, 1, 2);
    // L=0,I=0,O=0 -> idx 0
    // L=0,I=0,O=1 -> idx 1
    // L=1,I=0,O=0 -> idx 2
    // L=1,I=0,O=1 -> idx 3
    // ONNX [O, I, L]
    // O=0, I=0, L=0 -> src 0 (1.0)
    // O=0, I=0, L=1 -> src 2 (3.0)
    // O=1, I=0, L=0 -> src 1 (2.0)
    // O=1, I=0, L=1 -> src 3 (4.0)
    expect(Array.from(out)).toEqual([1.0, 3.0, 2.0, 4.0]);
  });

  it('transposeConv3DWeights', () => {
    const input = new Float32Array([1.0, 2.0]); // D=1, H=1, W=1, I=1, O=2
    const out = transposeConv3DWeights(input, 1, 1, 1, 1, 2);
    expect(Array.from(out)).toEqual([1.0, 2.0]);
  });

  it('transposeDenseWeights', () => {
    const input = new Float32Array([1.0, 2.0, 3.0, 4.0]); // 2x2
    const out = transposeDenseWeights(input, 2, 2);
    // [I, O] -> [O, I]
    // I0,O0 = 1.0 -> O0,I0 (idx 0)
    // I0,O1 = 2.0 -> O1,I0 (idx 2)
    // I1,O0 = 3.0 -> O0,I1 (idx 1)
    // I1,O1 = 4.0 -> O1,I1 (idx 3)
    expect(Array.from(out)).toEqual([1.0, 3.0, 2.0, 4.0]);
  });

  it('calculates padding', () => {
    expect(calculatePaddingSame(10, 3, 1, 1)).toEqual([1, 1]);
    expect(calculatePaddingSame(10, 4, 1, 1)).toEqual([1, 2]);
    expect(calculatePaddingValid()).toEqual([0, 0]);
  });
});
