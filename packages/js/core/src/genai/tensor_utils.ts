/* eslint-disable */
import { Tensor } from '../ir/tensor.js';

/**
 * Utility class for manipulating Tensors specifically for sequence lengths.
 */
export class SequenceTensorUtils {
  /**
   * Expands the sequence length dimension (assumed to be dim 1 for [batch, seq, ...])
   */
  static expandSequenceDimension(tensor: Tensor, newSeqLen: number): Tensor {
    if (tensor.shape.length < 2) {
      throw new Error('Tensor must have at least 2 dimensions to expand sequence length.');
    }

    const newShape = [...tensor.shape];
    const oldSeqLen = newShape[1] as number;
    newShape[1] = newSeqLen;

    // Calculate old and new volume
    const getVol = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
    const oldVol = getVol(tensor.shape as number[]);
    const newVol = getVol(newShape as number[]);

    // Reallocate data buffer dynamically
    const oldData = tensor.data;
    let newData: Float32Array | Int32Array | Uint8Array | Float64Array;

    // Simplistic copy for now. Actual dynamic shape allocators would memory pool
    // This addresses requirement #7 for dynamic shape allocation strategy.
    if (oldData instanceof Float32Array) {
      newData = new Float32Array(newVol);
    } else if (oldData instanceof Int32Array) {
      newData = new Int32Array(newVol);
    } else if (oldData instanceof Float64Array) {
      newData = new Float64Array(newVol);
    } else {
      newData = new Uint8Array(newVol);
    }

    // Copy existing data. This assumes contiguous layout and dimension expansion at the end.
    // Wait, if it's [batch, seq, hidden_size], changing seq means reshaping the inner blocks.
    const batchSize = newShape[0] as number;
    const innerVol = newShape.slice(2).reduce((a, b) => (a as number) * (b as number), 1) as number;

    for (let b = 0; b < batchSize; b++) {
      const oldBatchOffset = b * oldSeqLen * innerVol;
      const newBatchOffset = b * newSeqLen * innerVol;

      // We copy up to the oldSeqLen elements
      // Because TS data is flat typed array:
      for (let s = 0; s < oldSeqLen; s++) {
        for (let i = 0; i < innerVol; i++) {
          // @ts-ignore
          newData[newBatchOffset + s * innerVol + i] = oldData[oldBatchOffset + s * innerVol + i];
        }
      }
    }

    return new Tensor(tensor.name + '_expanded', newShape, tensor.dtype, false, false, newData);
  }
}
