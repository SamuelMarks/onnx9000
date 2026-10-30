/**
 * @fileoverview logger.ts
 * Provides logger functionality for the serve package.
 */
// 144. Allow granular control of logging levels
export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
}

/** Represents the `Logger` class. */
export class Logger {
  /** Property `level` of `Logger`. */
  public level: LogLevel = LogLevel.INFO;
  // 143. Support exporting logs to Datadog / NewRelic natively via HTTP POST.
  /** Property `exporterUrl` of `Logger`. */
  public exporterUrl?: string | undefined;

  /**
   * Initializes a new `Logger` instance.
   * @param level Parameter `level`.
   * @param exporterUrl Parameter `exporterUrl`.
   */
  constructor(level: LogLevel = LogLevel.INFO, exporterUrl?: string) {
    this.level = level;
    this.exporterUrl = exporterUrl;
  }

  /**
   * Executes `export` on `Logger`.
   * @param levelStr Parameter `levelStr`.
   * @param message Parameter `message`.
   * @param meta Parameter `meta`.
   * @returns Result of `export`.
   */
  private async export(levelStr: string, message: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.exporterUrl) {
      try {
        await fetch(this.exporterUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            timestamp: new Date().toISOString(),
            level: levelStr,
            message,
            ...meta,
          }),
        }).catch(() => undefined); // silent fail on export
      } catch {
        // Ignore export error
      }
    }
  }

  /**
   * Executes `trace` on `Logger`.
   * @param msg Parameter `msg`.
   * @param meta Parameter `meta`.
   * @returns Result of `trace`.
   */
  public trace(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.TRACE) {
      console.trace(msg, meta || '');
      this.export('TRACE', msg, meta);
    }
  }
  /**
   * Executes `debug` on `Logger`.
   * @param msg Parameter `msg`.
   * @param meta Parameter `meta`.
   * @returns Result of `debug`.
   */
  public debug(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.DEBUG) {
      console.debug(msg, meta || '');
      this.export('DEBUG', msg, meta);
    }
  }
  /**
   * Executes `info` on `Logger`.
   * @param msg Parameter `msg`.
   * @param meta Parameter `meta`.
   * @returns Result of `info`.
   */
  public info(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.INFO) {
      console.info(msg, meta || '');
      this.export('INFO', msg, meta);
    }
  }
  /**
   * Executes `warn` on `Logger`.
   * @param msg Parameter `msg`.
   * @param meta Parameter `meta`.
   * @returns Result of `warn`.
   */
  public warn(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.WARN) {
      console.warn(msg, meta || '');
      this.export('WARN', msg, meta);
    }
  }
  /**
   * Executes `error` on `Logger`.
   * @param msg Parameter `msg`.
   * @param meta Parameter `meta`.
   * @returns Result of `error`.
   */
  public error(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.ERROR) {
      console.error(msg, meta || '');
      this.export('ERROR', msg, meta);
    }
  }
}

export const globalLogger = new Logger();

// 141. Provide native OpenTelemetry traces (distributed tracing headers extraction).
/**
 * Executes `extractTraceContext`.
 * @param req Parameter `req`.
 * @returns Result of `extractTraceContext`.
 */
export function extractTraceContext(req: Request) {
  const traceparent = req.headers.get('traceparent');
  const tracestate = req.headers.get('tracestate');
  return { traceparent, tracestate };
}
