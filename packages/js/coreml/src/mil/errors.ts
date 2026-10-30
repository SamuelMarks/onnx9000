/**
 * @fileoverview errors.ts
 * Provides errors functionality for the coreml package.
 */
export class CoreMLExportError extends Error {
  /**
   * Initializes a new `CoreMLExportError` instance.
   * @param message Parameter `message`.
   */
  constructor(message: string) {
    super(message);
    this.name = 'CoreMLExportError';
  }
}

/** Represents the `UnsupportedOpError` class. */
export class UnsupportedOpError extends CoreMLExportError {
  /**
   * Initializes a new `UnsupportedOpError` instance.
   * @param opType Parameter `opType`.
   * @param reason Parameter `reason`.
   */
  constructor(opType: string, reason?: string) {
    super(`Unsupported ONNX operation: ${opType}${reason ? ` (${reason})` : ''}`);
    this.name = 'UnsupportedOpError';
  }
}

/** Represents the `ThermalThrottlingWarning` class. */
export class ThermalThrottlingWarning extends Error {
  /**
   * Initializes a new `ThermalThrottlingWarning` instance.
   * @param topology Parameter `topology`.
   */
  constructor(topology: string) {
    super(
      `Warning: The detected topology (${topology}) is known to potentially trigger ANE thermal throttling.`,
    );
    this.name = 'ThermalThrottlingWarning';
  }
}

/** Represents the `ANELimitsExceededWarning` class. */
export class ANELimitsExceededWarning extends Error {
  /**
   * Initializes a new `ANELimitsExceededWarning` instance.
   * @param reason Parameter `reason`.
   */
  constructor(reason: string) {
    super(`Warning: ANE physical limits exceeded: ${reason}`);
    this.name = 'ANELimitsExceededWarning';
  }
}

/** Represents the `DoubleDowncastWarning` class. */
export class DoubleDowncastWarning extends Error {
  /** Initializes a new `DoubleDowncastWarning` instance. */
  constructor() {
    super('Warning: float64 is not natively supported by MIL. Implicitly casting to float32.');
    this.name = 'DoubleDowncastWarning';
  }
}
