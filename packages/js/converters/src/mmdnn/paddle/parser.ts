/**
 * @fileoverview Parser for PaddlePaddle model definitions and binary weights.
 */
import type { PaddleModel } from '../types.js';

export type { PaddleModel };

/** Metadata returned when parsing a PaddlePaddle weights buffer. */
export interface PaddleWeightsInfo {
  /** Total byte length of the parsed weights buffer. */
  byteLength: number;
}

/**
 * Parser for PaddlePaddle models.
 */
export class PaddleParser {
  /**
   * Parses a PaddlePaddle model description from JSON text or a pre-parsed `PaddleModel`.
   * @param modelJson Model JSON string or `PaddleModel` structure.
   * @returns Parsed `PaddleModel` object.
   */
  public parseModel(modelJson: string | PaddleModel): PaddleModel {
    if (typeof modelJson === 'string') {
      try {
        return JSON.parse(modelJson) as PaddleModel;
      } catch {
        return { blocks: [] };
      }
    }
    return modelJson;
  }

  /**
   * Parses PaddlePaddle binary weights.
   * @param weightsBuffer Binary weights buffer.
   * @returns Metadata describing the parsed weights buffer.
   */
  public parseWeights(weightsBuffer: Uint8Array): PaddleWeightsInfo {
    return {
      byteLength: weightsBuffer.byteLength,
    };
  }
}
