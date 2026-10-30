/**
 * @fileoverview types.ts
 * Provides types functionality for the coreml package.
 */
export enum MILDataType {
  FLOAT32 = 'fp32',
  FLOAT16 = 'fp16',
  INT32 = 'int32',
  INT64 = 'int64',
  BOOL = 'bool',
  STRING = 'string',
}

/** Represents the `MILType` class. */
export abstract class MILType {
  /**
   * Executes `isTensor` on `MILType`.
   * @returns Result of `isTensor`.
   */
  abstract isTensor(): boolean;
  /**
   * Executes `isScalar` on `MILType`.
   * @returns Result of `isScalar`.
   */
  abstract isScalar(): boolean;
  /**
   * Executes `isTuple` on `MILType`.
   * @returns Result of `isTuple`.
   */
  abstract isTuple(): boolean;
  /**
   * Executes `toString` on `MILType`.
   * @returns Result of `toString`.
   */
  abstract toString(): string;
}

/** Represents the `TensorType` class. */
export class TensorType extends MILType {
  /**
   * Initializes a new `TensorType` instance.
   * @param dataType Parameter `dataType`.
   * @param shape Parameter `shape`.
   */
  constructor(
    public dataType: MILDataType,
    public shape: (number | string)[],
  ) {
    super();
  }
  /**
   * Executes `isTensor` on `TensorType`.
   * @returns Result of `isTensor`.
   */
  isTensor() {
    return true;
  }
  /**
   * Executes `isScalar` on `TensorType`.
   * @returns Result of `isScalar`.
   */
  isScalar() {
    return false;
  }
  /**
   * Executes `isTuple` on `TensorType`.
   * @returns Result of `isTuple`.
   */
  isTuple() {
    return false;
  }
  /**
   * Executes `toString` on `TensorType`.
   * @returns Result of `toString`.
   */
  toString() {
    return `tensor<${this.dataType}, [${this.shape.join(',')}]>`;
  }
}

/** Represents the `ScalarType` class. */
export class ScalarType extends MILType {
  /**
   * Initializes a new `ScalarType` instance.
   * @param dataType Parameter `dataType`.
   */
  constructor(public dataType: MILDataType) {
    super();
  }
  /**
   * Executes `isTensor` on `ScalarType`.
   * @returns Result of `isTensor`.
   */
  isTensor() {
    return false;
  }
  /**
   * Executes `isScalar` on `ScalarType`.
   * @returns Result of `isScalar`.
   */
  isScalar() {
    return true;
  }
  /**
   * Executes `isTuple` on `ScalarType`.
   * @returns Result of `isTuple`.
   */
  isTuple() {
    return false;
  }
  /**
   * Executes `toString` on `ScalarType`.
   * @returns Result of `toString`.
   */
  toString() {
    return this.dataType;
  }
}

/** Represents the `TupleType` class. */
export class TupleType extends MILType {
  /**
   * Initializes a new `TupleType` instance.
   * @param elements Parameter `elements`.
   */
  constructor(public elements: MILType[]) {
    super();
  }
  /**
   * Executes `isTensor` on `TupleType`.
   * @returns Result of `isTensor`.
   */
  isTensor() {
    return false;
  }
  /**
   * Executes `isScalar` on `TupleType`.
   * @returns Result of `isScalar`.
   */
  isScalar() {
    return false;
  }
  /**
   * Executes `isTuple` on `TupleType`.
   * @returns Result of `isTuple`.
   */
  isTuple() {
    return true;
  }
  /**
   * Executes `toString` on `TupleType`.
   * @returns Result of `toString`.
   */
  toString() {
    return `tuple<${this.elements.map((e) => e.toString()).join(', ')}>`;
  }
}
