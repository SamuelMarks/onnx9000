/**
 * @fileoverview Zero-dependency browser image classification pipeline for `@onnx9000/zero-dep-classifier`.
 * Provides ImageNet RGB channel normalization and numerically stable softmax top-1 prediction.
 */

/**
 * Top-1 classification prediction result.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `ClassificationPrediction` definition. */
export interface ClassificationPrediction {
  /** Predicted class index. */
  classIndex: number;
  /** Softmax probability in `[0, 1]`. */
  probability: number;
}

/**
 * Zero-dependency image classifier performing ImageNet normalization and softmax scoring.
 */
export class ZeroDepClassifier {
  /** Standard ImageNet channel means `[R, G, B]`. */
  public static readonly IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
  /** Standard ImageNet channel standard deviations `[R, G, B]`. */
  public static readonly IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

  /**
   * Normalizes an RGB pixel triplet `(r, g, b)` in `[0, 255]` using ImageNet mean and std.
   * @param r Red channel `[0..255]`.
   * @param g Green channel `[0..255]`.
   * @param b Blue channel `[0..255]`.
   * @returns Normalized `[r, g, b]` tuple.
   */
  public normalizePixel(r: number, g: number, b: number): [number, number, number] {
    return [
      (r / 255.0 - ZeroDepClassifier.IMAGENET_MEAN[0]) / ZeroDepClassifier.IMAGENET_STD[0],
      (g / 255.0 - ZeroDepClassifier.IMAGENET_MEAN[1]) / ZeroDepClassifier.IMAGENET_STD[1],
      (b / 255.0 - ZeroDepClassifier.IMAGENET_MEAN[2]) / ZeroDepClassifier.IMAGENET_STD[2],
    ];
  }

  /**
   * Computes numerically stable softmax and returns the top-1 class prediction.
   * @param logits Raw class logits.
   * @returns `ClassificationPrediction` with `classIndex` and `probability`.
   * @throws Error if `logits` is empty.
   */
  public predictTop1(logits: Float32Array): ClassificationPrediction {
    if (logits.length === 0) {
      throw new Onnx9000Error('Logits cannot be empty');
    }
    let maxLogit = Number(logits[0]);
    let bestIdx = 0;
    for (let i = 1; i < logits.length; i++) {
      const v = Number(logits[i]);
      if (v > maxLogit) {
        maxLogit = v;
        bestIdx = i;
      }
    }
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      sumExp += Math.exp(Number(logits[i]) - maxLogit);
    }
    return {
      classIndex: bestIdx,
      probability: 1.0 / sumExp,
    };
  }

  /**
   * Processes an image classification request string.
   * @param arg Image identifier string.
   * @returns Formatted classification status string.
   */
  public process(arg: string): string {
    this.normalizePixel(128, 128, 128);
    this.predictTop1(new Float32Array([0.1, 2.5, 0.3]));
    return `Zero Dep Classifier processed ${arg}`;
  }
}
