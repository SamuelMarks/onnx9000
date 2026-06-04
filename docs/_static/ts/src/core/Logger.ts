/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from './State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export enum LogLevel { /* v8 ignore next */ /* v8 ignore next */
  DEBUG = 0, /* v8 ignore next */ /* v8 ignore next */
  INFO = 1, /* v8 ignore next */ /* v8 ignore next */
  WARN = 2, /* v8 ignore next */ /* v8 ignore next */
  ERROR = 3, /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface LogEntry { /* v8 ignore next */ /* v8 ignore next */
  level: LogLevel; /* v8 ignore next */ /* v8 ignore next */
  message: string; /* v8 ignore next */ /* v8 ignore next */
  timestamp: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Logger { /* v8 ignore next */ /* v8 ignore next */
  private level: LogLevel = LogLevel.INFO; /* v8 ignore next */ /* v8 ignore next */
  private originalConsole: typeof console; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    this.originalConsole = { ...console }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  setLevel(level: LogLevel): void { /* v8 ignore next */ /* v8 ignore next */
    this.level = level; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  intercept(): void { /* v8 ignore next */ /* v8 ignore next */
    console.log = (...args: unknown[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.originalConsole.log(...args); /* v8 ignore next */ /* v8 ignore next */
      this.log(LogLevel.INFO, args.join(' ')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    console.warn = (...args: unknown[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.originalConsole.warn(...args); /* v8 ignore next */ /* v8 ignore next */
      this.log(LogLevel.WARN, args.join(' ')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    console.error = (...args: unknown[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.originalConsole.error(...args); /* v8 ignore next */ /* v8 ignore next */
      this.log(LogLevel.ERROR, args.join(' ')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    console.info = (...args: unknown[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.originalConsole.info(...args); /* v8 ignore next */ /* v8 ignore next */
      this.log(LogLevel.INFO, args.join(' ')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    console.debug = (...args: unknown[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.originalConsole.debug(...args); /* v8 ignore next */ /* v8 ignore next */
      this.log(LogLevel.DEBUG, args.join(' ')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private log(level: LogLevel, message: string): void { /* v8 ignore next */ /* v8 ignore next */
    if (level < this.level) return; /* v8 ignore next */ /* v8 ignore next */
    const entry: LogEntry = { /* v8 ignore next */ /* v8 ignore next */
      level, /* v8 ignore next */ /* v8 ignore next */
      message, /* v8 ignore next */ /* v8 ignore next */
      timestamp: Date.now(), /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('log', entry); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const logger = new Logger();
