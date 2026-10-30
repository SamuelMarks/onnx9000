/**
 * @fileoverview reporter.ts
 * Provides reporter functionality for the converters package.
 */
export class MMDNNError extends Error {
  /**
   * Initializes a new `MMDNNError` instance.
   * @param message Parameter `message`.
   */
  constructor(message: string) {
    super(message);
    this.name = 'MMDNNError';
  }
}

/** Represents the `MMDNNReporter` class. */
export class MMDNNReporter {
  /** Property `warnings` of `MMDNNReporter`. */
  warnings: string[] = [];
  /** Property `errors` of `MMDNNReporter`. */
  errors: string[] = [];
  /** Property `logs` of `MMDNNReporter`. */
  logs: string[] = [];
  /** Property `verbose` of `MMDNNReporter`. */
  verbose: boolean;

  /**
   * Initializes a new `MMDNNReporter` instance.
   * @param verbose Parameter `verbose`.
   */
  constructor(verbose: boolean = false) {
    this.verbose = verbose;
  }

  /**
   * Executes `info` on `MMDNNReporter`.
   * @param msg Parameter `msg`.
   */
  info(msg: string): void {
    const formatted = `[INFO] ${msg}`;
    this.logs.push(formatted);
    if (this.verbose) {
      console.log(formatted);
    }
  }

  /**
   * Executes `warn` on `MMDNNReporter`.
   * @param msg Parameter `msg`.
   * @param nodeName Parameter `nodeName`.
   */
  warn(msg: string, nodeName?: string): void {
    const prefix = nodeName ? `[WARN] (Node: ${nodeName}):` : '[WARN]:';
    const formatted = `${prefix} ${msg}`;
    this.warnings.push(formatted);
    if (this.verbose) {
      console.warn(formatted);
    }
  }

  /**
   * Executes `error` on `MMDNNReporter`.
   * @param msg Parameter `msg`.
   * @param nodeName Parameter `nodeName`.
   * @returns Result of `error`.
   */
  error(msg: string, nodeName?: string): object {
    const prefix = nodeName ? `[ERROR] (Node: ${nodeName}):` : '[ERROR]:';
    const formatted = `${prefix} ${msg}`;
    this.errors.push(formatted);
    if (this.verbose) {
      console.error(formatted);
      console.trace();
    }
    throw new MMDNNError(formatted);
  }

  /**
   * Executes `getReport` on `MMDNNReporter`.
   * @returns Result of `getReport`.
   */
  getReport(): string {
    return [
      '--- MMDNN Conversion Report ---',
      `Logs: ${this.logs.length}`,
      `Warnings: ${this.warnings.length}`,
      `Errors: ${this.errors.length}`,
      '',
      ...this.warnings,
      ...this.errors,
    ].join('\n');
  }
}
