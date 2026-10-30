/**
 * @fileoverview Unified custom error hierarchy for the ONNX9000 TypeScript ecosystem.
 * Provides structured domain error classes extending `Onnx9000Error`.
 */

/**
 * Base class for all errors thrown across ONNX9000 packages.
 */
export class Onnx9000Error extends Error {
  /**
   * Creates a new `Onnx9000Error`.
   * @param message Human-readable error message.
   */
  constructor(message = 'An error occurred') {
    super(message);
    this.name = new.target.name;
  }
}

/**
 * Error thrown when parsing an ONNX Protobuf or binary model fails.
 */
export class ONNXParseError extends Onnx9000Error {}

/**
 * Error thrown when static or symbolic shape inference fails.
 */
export class ShapeInferenceError extends Onnx9000Error {}

/**
 * Error thrown when an operator is unsupported by a target backend or converter.
 */
export class UnsupportedOpError extends Onnx9000Error {
  /** The name of the unsupported operator. */
  public readonly opType: string;

  /**
   * Creates a new `UnsupportedOpError`.
   * @param opType Name of the unsupported operator.
   * @param message Optional custom error message.
   */
  constructor(opType: string, message = '') {
    super(message || `Operator '${opType}' is not supported yet.`);
    this.opType = opType;
  }
}

/**
 * Error thrown when converting a model from a source framework fails.
 */
export class ConversionError extends Onnx9000Error {}

/**
 * Error thrown when executing a graph on a hardware backend fails.
 */
export class BackendError extends Onnx9000Error {}
