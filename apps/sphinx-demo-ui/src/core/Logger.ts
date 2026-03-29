/* eslint-disable */
// @ts-nocheck
import { globalEventBus } from './EventBus';

export enum LogLevel {
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error'
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: Date;
}

/**
 * Utility to intercept console logs and emit them over the EventBus.
 * Used to pipe WASM and worker logs into the UI Console component.
 */
export class Logger {
  private static instance: Logger;
  private originalLog: typeof console.log;
  private originalWarn: typeof console.warn;
  private originalError: typeof console.error;
  private isIntercepting = false;

  private constructor() {
    this.originalLog = console.log.bind(console);
    this.originalWarn = console.warn.bind(console);
    this.originalError = console.error.bind(console);
  }

  public static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  /**
   * Starts intercepting `console.log`, `console.warn`, and `console.error`.
   */
  public startIntercepting(): void {
    if (this.isIntercepting) return;
    this.isIntercepting = true;

    console.log = (...args: object[]) => {
      this.originalLog(...args);
      this.emitLog(LogLevel.INFO, args);
    };

    console.warn = (...args: object[]) => {
      this.originalWarn(...args);
      this.emitLog(LogLevel.WARN, args);
    };

    console.error = (...args: object[]) => {
      this.originalError(...args);
      this.emitLog(LogLevel.ERROR, args);
    };
  }

  /**
   * Stops intercepting logs and restores the original console methods.
   */
  public stopIntercepting(): void {
    if (!this.isIntercepting) return;
    this.isIntercepting = false;

    console.log = this.originalLog;
    console.warn = this.originalWarn;
    console.error = this.originalError;
  }

  /**
   * Formats and emits a log entry via the global EventBus.
   */
  private emitLog(level: LogLevel, args: object[]): void {
    const message = args
      .map((arg) => {
        if (typeof arg === 'object') {
          try {
            return JSON.stringify(arg, null, 2);
          } catch {
            return String(arg);
          }
        }
        return String(arg);
      })
      .join(' ');

    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date()
    };

    globalEventBus.emit('CONSOLE_LOG', entry);
  }

  /**
   * Manually log a message directly into the stream without intercepting `console`.
   */
  public logDirect(level: LogLevel, message: string): void {
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date()
    };
    globalEventBus.emit('CONSOLE_LOG', entry);
  }
}
