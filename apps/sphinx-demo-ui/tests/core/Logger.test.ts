import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Logger, LogLevel, LogEntry } from '../../src/core/Logger';
import { globalEventBus } from '../../src/core/EventBus';

describe('Logger', () => {
  let emittedLogs: LogEntry[] = [];

  beforeEach(() => {
    emittedLogs = [];
    globalEventBus.on<LogEntry>('CONSOLE_LOG', (entry) => {
      emittedLogs.push(entry);
    });
    // Ensure we start from a clean non-intercepting state
    Logger.getInstance().stopIntercepting();
  });

  afterEach(() => {
    Logger.getInstance().stopIntercepting();
    globalEventBus.clearAll();
  });

  it('should be a singleton', () => {
    const l1 = Logger.getInstance();
    const l2 = Logger.getInstance();
    expect(l1).toBe(l2);
  });

  it('should intercept console.log, warn, and error', () => {
    const logger = Logger.getInstance();

    // Silence the actual console for this test to avoid test output noise
    const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    logger.startIntercepting();

    console.log('hello world', 42);
    console.warn('a warning');
    console.error('an error', { obj: true });

    expect(emittedLogs.length).toBe(3);

    expect(emittedLogs[0].level).toBe(LogLevel.INFO);
    expect(emittedLogs[0].message).toBe('hello world 42');

    expect(emittedLogs[1].level).toBe(LogLevel.WARN);
    expect(emittedLogs[1].message).toBe('a warning');

    expect(emittedLogs[2].level).toBe(LogLevel.ERROR);
    expect(emittedLogs[2].message).toMatch(/an error/);
    expect(emittedLogs[2].message).toMatch(/"obj": true/);

    // Call startIntercepting again (should no-op)
    logger.startIntercepting();

    logger.stopIntercepting();
    consoleLogSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  it('should handle JSON stringify circular reference failures gracefully', () => {
    const logger = Logger.getInstance();

    const objA: any = {};
    const objB: any = { a: objA };
    objA.b = objB; // circular

    const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    logger.startIntercepting();
    console.log(objA);
    logger.stopIntercepting();

    expect(emittedLogs.length).toBe(1);
    expect(emittedLogs[0].message).toBe('[object Object]'); // Fallback to String(arg)

    consoleLogSpy.mockRestore();
  });

  it('should log directly without interception', () => {
    const logger = Logger.getInstance();
    logger.logDirect(LogLevel.INFO, 'direct message');

    expect(emittedLogs.length).toBe(1);
    expect(emittedLogs[0].message).toBe('direct message');
  });

  it('should stop intercepting correctly', () => {
    const logger = Logger.getInstance();
    const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    logger.startIntercepting();
    logger.stopIntercepting();
    // Call twice to hit the early return
    logger.stopIntercepting();

    console.log('should not be emitted');

    expect(emittedLogs.length).toBe(0);

    consoleLogSpy.mockRestore();
  });
});
