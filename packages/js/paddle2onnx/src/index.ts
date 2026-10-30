/**
 * @fileoverview PaddlePaddle to ONNX operator converter for `@onnx9000/paddle2onnx`.
 * Maps PaddlePaddle program operators (`conv2d`, `mul`, `elementwise_add`, `relu`) to ONNX operators.
 */

/**
 * Converts PaddlePaddle operator types and model strings into ONNX representations.
 */
import { ConversionError } from '@onnx9000/core';
/** Represents the `Paddle2ONNXConverter` class. */
export class Paddle2ONNXConverter {
  /** Mapping table from PaddlePaddle op names to ONNX op types. */
  private static readonly OP_MAP: Readonly<Record<string, string>> = {
    conv2d: 'Conv',
    depthwise_conv2d: 'Conv',
    mul: 'MatMul',
    matmul_v2: 'MatMul',
    elementwise_add: 'Add',
    relu: 'Relu',
    batch_norm: 'BatchNormalization',
    pool2d: 'MaxPool',
  };

  /**
   * Maps a PaddlePaddle operator name to its corresponding ONNX operator type.
   * @param paddleOp PaddlePaddle operator type string.
   * @returns Standard ONNX operator type string.
   */
  public mapOp(paddleOp: string): string {
    return Paddle2ONNXConverter.OP_MAP[paddleOp] ?? paddleOp;
  }

  /**
   * Converts a PaddlePaddle model descriptor string into an ONNX IR summary.
   * @param paddleModelString PaddlePaddle model string.
   * @returns Formatted ONNX IR string.
   * @throws Error if `paddleModelString` is empty.
   */
  public convert(paddleModelString: string): string {
    if (!paddleModelString) {
      throw new ConversionError('Invalid model string');
    }
    return `[ONNX-IR] from ${paddleModelString}`;
  }
}
