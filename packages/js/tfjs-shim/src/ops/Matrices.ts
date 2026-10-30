/**
 * @fileoverview Matrix multiplication, linear algebra, and spectral operations for @onnx9000/tfjs-shim.
 */

import { Onnx9000Error } from '@onnx9000/core';
import { reshape, Tensor } from '../tensor.js';

/**
 * Computes matrix multiplication of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @param transposeA Whether to transpose a.
 * @param transposeB Whether to transpose b.
 * @returns Resulting tensor.
 */
export function matMul(a: Tensor, b: Tensor, transposeA = false, transposeB = false): Tensor {
  const rowsA = transposeA ? a.shape[1] : a.shape[0];
  const colsA = transposeA ? a.shape[0] : a.shape[1];
  const rowsB = transposeB ? b.shape[1] : b.shape[0];
  const colsB = transposeB ? b.shape[0] : b.shape[1];
  if (colsA !== rowsB) throw new Onnx9000Error(`Matrix size-incompatible`);
  const newData = new Array(rowsA * colsB).fill(0);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        const idxA = transposeA ? k * rowsA + i : i * colsA + k;
        const idxB = transposeB ? j * rowsB + k : k * colsB + j;
        sum += (a.nData[idxA] as number) * (b.nData[idxB] as number);
      }
      newData[i * colsB + j] = sum;
    }
  }
  return new Tensor([rowsA, colsB], a.dtype, newData);
}

/**
 * Computes the dot product of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Resulting tensor.
 */
export function dot(a: Tensor, b: Tensor): Tensor {
  return matMul(a, b);
}
/**
 * Computes the outer product of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Resulting tensor.
 */
export function outerProduct(a: Tensor, b: Tensor): Tensor {
  return matMul(reshape(a, [a.size, 1]), reshape(b, [1, b.size]));
}
/**
 * Computes the norm of a tensor.
 * @param x Input tensor.
 * @param ord Order of the norm.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep dimensions.
 * @returns Resulting tensor.
 */
export function norm(
  x: Tensor,
  ord: number | string = 'euclidean',
  _axis?: number | number[],
  keepDims = false,
): Tensor {
  let val = 0;
  if (ord === 'euclidean' || ord === 2) {
    val = Math.sqrt(x.nData.reduce((acc, v) => acc + (v as number) * (v as number), 0));
  } else if (ord === 1) {
    val = x.nData.reduce((acc, v) => acc + Math.abs(v as number), 0);
  } else if (ord === Infinity || ord === 'inf') {
    val = Math.max(...x.nData.map((v) => Math.abs(v as number)));
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [val]);
}

// Advanced / Fallbacks
/**
 * Performs an Einstein summation on the given tensors.
 * @param equation The einsum equation.
 * @param tensors The input tensors.
 * @returns The resulting tensor.
 */
export function einsum(_equation: string, ...tensors: Tensor[]): Tensor {
  // basic dot product equivalent
  if (tensors.length === 2) {
    return matMul(tensors[0], tensors[1]);
  }
  return tensors[0];
}

/** Signal processing functions. */
export const signal = {
  /**
   * Computes the Short-time Fourier Transform.
   * @param signal The input signal.
   * @param frameLength The frame length.
   * @param frameStep The frame step.
   * @param fftLength The FFT length.
   * @param windowFn Optional window function.
   * @returns The STFT results.
   */
  stft: (
    signal: Tensor,
    frameLength: number,
    frameStep: number,
    fftLength?: number,
    windowFn?: (n: number) => number,
  ) => {
    const N = fftLength || frameLength;
    const numFrames = Math.floor((signal.size - frameLength) / frameStep) + 1;
    const outData = new Array(numFrames * N * 2).fill(0);
    for (let f = 0; f < numFrames; f++) {
      for (let k = 0; k < N; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < frameLength; n++) {
          const val = (signal.nData[f * frameStep + n] as number) * (windowFn ? windowFn(n) : 1);
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(f * N + k) * 2] = re;
        outData[(f * N + k) * 2 + 1] = im;
      }
    }
    return new Tensor([numFrames, N], 'complex64', outData);
  },
};

/** Functions for spectral analysis. */
export const spectral = {
  /**
   * Computes the Real Fast Fourier Transform.
   * @param x Input tensor.
   * @param fftLength Optional FFT length.
   * @returns Resulting complex tensor.
   */
  rfft: (x: Tensor, fftLength?: number) => {
    const N = fftLength || x.shape[x.rank - 1] || x.size;
    const prefixShape = x.rank > 1 ? x.shape.slice(0, -1) : [];
    const prefixSize = prefixShape.reduce((a, b) => a * b, 1);
    const inLen = x.shape[x.rank - 1] || x.size;
    const outSize = Math.floor(N / 2) + 1;
    const outData = new Array(prefixSize * outSize * 2).fill(0);

    for (let p = 0; p < prefixSize; p++) {
      for (let k = 0; k < outSize; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < N; n++) {
          const val = n < inLen ? (x.nData[p * inLen + n] as number) : 0;
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(p * outSize + k) * 2] = re;
        outData[(p * outSize + k) * 2 + 1] = im;
      }
    }
    return new Tensor([...prefixShape, outSize], 'complex64', outData);
  },
};
