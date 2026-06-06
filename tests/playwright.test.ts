import { describe, expect, it } from 'vitest';
import config from './playwright.config.js';

describe('playwright.config', () => {
  it('should export config', () => {
    expect(config).toBeDefined();
  });
});
