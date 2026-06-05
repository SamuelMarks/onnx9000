// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { Logger, LogLevel } from '../src/core/Logger.js';
import { globalEventBus } from '../src/core/EventBus.js';

describe('Logger', () => {
  it('should intercept and emit logs', () => {
    const logger = Logger.getInstance();
    logger.startIntercepting();

    let caughtLog = false;
    globalEventBus.on('CONSOLE_LOG', (entry: any) => {
      if (entry.message.includes('test_intercept')) {
        caughtLog = true;
      }
    });

    console.log('test_intercept');
    expect(caughtLog).toBe(true);

    logger.stopIntercepting();
  });
});
