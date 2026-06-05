import { describe, it, expect } from 'vitest';
import config from './playwright-native.config.js';

describe('playwright-native.config', () => {
  it('should export config', () => {
    expect(config).toBeDefined();
  });
});
