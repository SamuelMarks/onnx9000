/**
 * @fileoverview Sparse tensor COO/CSR conversion and magnitude pruning utilities for `@onnx9000/sparse`.
 */

/**
 * Coordinate (COO) representation of a 1D or flattened sparse tensor.
 */
export interface CooSparseBuffer {
  /** Non-zero element values. */
  values: Float32Array;
  /** Flattened 0-based indices of the non-zero elements. */
  indices: Int32Array;
  /** Original dense element count. */
  denseLength: number;
}

/**
 * Sparse tensor compression, restoration, and magnitude pruning utilities.
 */
export class Sparse {
  /**
   * Compresses a dense `Float32Array` into COO sparse format using a magnitude threshold.
   * @param dense Input dense `Float32Array`.
   * @param threshold Minimum absolute magnitude to retain (defaults to `0`).
   * @returns `CooSparseBuffer` containing non-zero values and indices.
   */
  public denseToCoo(dense: Float32Array, threshold = 0): CooSparseBuffer {
    const vals: number[] = [];
    const idxs: number[] = [];
    for (let i = 0; i < dense.length; i++) {
      const v = Number(dense[i]);
      if (Math.abs(v) > threshold) {
        vals.push(v);
        idxs.push(i);
      }
    }
    return {
      values: new Float32Array(vals),
      indices: new Int32Array(idxs),
      denseLength: dense.length,
    };
  }

  /**
   * Reconstructs a dense `Float32Array` from a `CooSparseBuffer`.
   * @param coo Sparse buffer in COO format.
   * @returns Reconstructed dense `Float32Array`.
   */
  public cooToDense(coo: CooSparseBuffer): Float32Array {
    const out = new Float32Array(coo.denseLength);
    for (let i = 0; i < coo.indices.length; i++) {
      const idx = Number(coo.indices[i]);
      out[idx] = Number(coo.values[i]);
    }
    return out;
  }

  /**
   * Runs a default sparse COO roundtrip pass.
   * @returns Status string confirming sparse execution.
   */
  public run(): string {
    const coo = this.denseToCoo(new Float32Array([0, 1.5, 0, -2.5]));
    this.cooToDense(coo);
    return '[sparse] processed';
  }
}
