import { describe, expect, it } from 'vitest';
import config from './playwright-native.config.js';

describe('playwright-native.config', () => {
  it('should export config', () => {
    expect(config).toBeDefined();
  });
});
