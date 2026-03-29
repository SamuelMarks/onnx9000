/* eslint-disable */
// @ts-nocheck
export class MMDNNError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MMDNNError';
  }
}

export class MMDNNReporter {
  warnings: string[] = [];
  errors: string[] = [];
  logs: string[] = [];
  verbose: boolean;

  constructor(verbose: boolean = false) {
    this.verbose = verbose;
  }

  info(msg: string): void {
    const formatted = `[INFO] ${msg}`;
    this.logs.push(formatted);
    if (this.verbose) {
      console.log(formatted);
    }
  }

  warn(msg: string, nodeName?: string): void {
    const prefix = nodeName ? `[WARN] (Node: ${nodeName}):` : '[WARN]:';
    const formatted = `${prefix} ${msg}`;
    this.warnings.push(formatted);
    if (this.verbose) {
      console.warn(formatted);
    }
  }

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
