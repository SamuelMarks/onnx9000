export class CoreMLExportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CoreMLExportError';
  }
}

export class UnsupportedOpError extends CoreMLExportError {
  constructor(opType: string, reason?: string) {
    super(`Unsupported ONNX operation: ${opType}${reason ? ` (${reason})` : ''}`);
    this.name = 'UnsupportedOpError';
  }
}

export class ThermalThrottlingWarning extends Error {
  constructor(topology: string) {
    super(
      `Warning: The detected topology (${topology}) is known to potentially trigger ANE thermal throttling.`,
    );
    this.name = 'ThermalThrottlingWarning';
  }
}

export class ANELimitsExceededWarning extends Error {
  constructor(reason: string) {
    super(`Warning: ANE physical limits exceeded: ${reason}`);
    this.name = 'ANELimitsExceededWarning';
  }
}

export class DoubleDowncastWarning extends Error {
  constructor() {
    super('Warning: float64 is not natively supported by MIL. Implicitly casting to float32.');
    this.name = 'DoubleDowncastWarning';
  }
}
