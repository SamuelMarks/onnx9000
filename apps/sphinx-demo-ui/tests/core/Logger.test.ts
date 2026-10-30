import { describe, expect, it } from 'vitest';
import { Logger } from '../../src/core/Logger';

describe('Logger.ts', () => {
  it('should instantiate and cover Logger', () => {
    const obj = Logger.getInstance();
    expect(obj).toBeDefined();
  });
});
