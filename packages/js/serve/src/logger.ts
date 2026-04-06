// 144. Allow granular control of logging levels
export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
}

export class Logger {
  public level: LogLevel = LogLevel.INFO;
  // 143. Support exporting logs to Datadog / NewRelic natively via HTTP POST.
  public exporterUrl?: string | undefined;

  constructor(level: LogLevel = LogLevel.INFO, exporterUrl?: string) {
    this.level = level;
    this.exporterUrl = exporterUrl;
  }

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
        }).catch(() => {}); // silent fail on export
      } catch (err) {}
    }
  }

  public trace(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.TRACE) {
      console.trace(msg, meta || '');
      this.export('TRACE', msg, meta);
    }
  }
  public debug(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.DEBUG) {
      console.debug(msg, meta || '');
      this.export('DEBUG', msg, meta);
    }
  }
  public info(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.INFO) {
      console.info(msg, meta || '');
      this.export('INFO', msg, meta);
    }
  }
  public warn(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.WARN) {
      console.warn(msg, meta || '');
      this.export('WARN', msg, meta);
    }
  }
  public error(msg: string, meta?: ReturnType<typeof JSON.parse>) {
    if (this.level <= LogLevel.ERROR) {
      console.error(msg, meta || '');
      this.export('ERROR', msg, meta);
    }
  }
}

export const globalLogger = new Logger();

// 141. Provide native OpenTelemetry traces (distributed tracing headers extraction).
export function extractTraceContext(req: Request) {
  /* v8 ignore start */
  const traceparent = req.headers.get('traceparent');
  const tracestate = req.headers.get('tracestate');
  return { traceparent, tracestate };
}
/* v8 ignore stop */
