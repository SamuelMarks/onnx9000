// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { globalEventBus } from '../src/core/EventBus.js';
import { Logger } from '../src/core/Logger.js';

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
