// @ts-nocheck
import { describe, it, expect } from 'vitest';
import config from '../playwright.config.js';

describe('playwright config', () => {
  it('should export config', () => {
    expect(config).toBeDefined();
    expect(config.testDir).toBe('./e2e');
  });
});
