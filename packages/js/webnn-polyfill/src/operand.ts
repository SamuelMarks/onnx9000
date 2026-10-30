/**
 * @fileoverview operand.ts
 * Provides operand functionality for the webnn-polyfill package.
 */
import type { MLOperand, MLOperandDataType } from './interfaces.js';

/** Represents the `PolyfillMLOperand` class. */
export class PolyfillMLOperand implements MLOperand {
  /** Property `dataType` of `PolyfillMLOperand`. */
  dataType: MLOperandDataType;
  /** Property `shape` of `PolyfillMLOperand`. */
  shape: number[];
  /** Property `name` of `PolyfillMLOperand`. */
  name: string; // The ONNX value name

  /**
   * Initializes a new `PolyfillMLOperand` instance.
   * @param name Parameter `name`.
   * @param dataType Parameter `dataType`.
   * @param shape Parameter `shape`.
   */
  constructor(name: string, dataType: MLOperandDataType, shape: number[]) {
    this.name = name;
    this.dataType = dataType;
    this.shape = shape;
  }
}
