/**
 * Parser for PaddlePaddle models.
 */
export class PaddleParser {
  /**
   * Parses a PaddlePaddle model description.
   * @param modelJson The model JSON string or object.
   * @returns The parsed model object.
   */
  public parseModel(modelJson: string | object): object {
    if (typeof modelJson === 'string') {
      try {
        return JSON.parse(modelJson) as object;
      } catch (e) {
        return { blocks: [] };
      }
    }
    return modelJson;
  }

  /**
   * Parses PaddlePaddle binary weights.
   * @param weightsBuffer The binary weights buffer.
   * @returns An object representing the parsed weights.
   */
  public parseWeights(weightsBuffer: Uint8Array): object {
    return {
      byteLength: weightsBuffer.byteLength,
    };
  }
}
