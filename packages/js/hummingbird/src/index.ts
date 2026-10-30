/**
 * @fileoverview Traditional ML tree-to-GEMM transpiler for `@onnx9000/hummingbird`.
 * Lowers decision tree ensembles into matrix multiplications and elementwise comparisons.
 */

/**
 * Decision node in a binary decision tree.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `DecisionTreeNode` definition. */
export interface DecisionTreeNode {
  /** Feature column index tested at this split (-1 for leaf). */
  featureIndex: number;
  /** Split threshold value (or leaf prediction value when featureIndex < 0). */
  threshold: number;
}

/**
 * Compiled GEMM-based representation of a decision tree (Perfect Tree strategy).
 */
export interface GemmTreeWeights {
  /** Number of input features. */
  numFeatures: number;
  /** Number of internal split nodes. */
  numInternalNodes: number;
  /** Number of leaf nodes. */
  numLeaves: number;
  /** Flattened feature selection matrix `A` of shape `[numFeatures, numInternalNodes]`. */
  selectionMatrix: Float32Array;
  /** Split threshold vector `B` of shape `[numInternalNodes]`. */
  thresholds: Float32Array;
}

/**
 * Transpiles decision trees into dense matrix multiplication (GEMM) weights.
 */
export class Hummingbird {
  /**
   * Compiles a list of decision tree split nodes into GEMM tensors.
   * @param numFeatures Total number of input features.
   * @param nodes List of tree nodes (internal nodes have `featureIndex >= 0`).
   * @returns Compiled `GemmTreeWeights` matrices.
   * @throws Error if `numFeatures` is not positive.
   */
  public compileTreeToGemm(numFeatures: number, nodes: DecisionTreeNode[]): GemmTreeWeights {
    if (numFeatures <= 0) {
      throw new Onnx9000Error('numFeatures must be positive');
    }
    const internal = nodes.filter((n) => n.featureIndex >= 0 && n.featureIndex < numFeatures);
    const numInternalNodes = internal.length;
    const numLeaves = Math.max(1, numInternalNodes + 1);
    const selectionMatrix = new Float32Array(numFeatures * numInternalNodes);
    const thresholds = new Float32Array(numInternalNodes);

    for (let j = 0; j < numInternalNodes; j++) {
      const node = internal[j];
      if (node) {
        selectionMatrix[node.featureIndex * numInternalNodes + j] = 1.0;
        thresholds[j] = node.threshold;
      }
    }

    return {
      numFeatures,
      numInternalNodes,
      numLeaves,
      selectionMatrix,
      thresholds,
    };
  }

  /**
   * Runs a default tree-to-GEMM compilation pass.
   * @returns Status string confirming Hummingbird execution.
   */
  public run(): string {
    this.compileTreeToGemm(4, [
      { featureIndex: 0, threshold: 0.5 },
      { featureIndex: -1, threshold: 1.0 },
    ]);
    return '[hummingbird] processed';
  }
}
