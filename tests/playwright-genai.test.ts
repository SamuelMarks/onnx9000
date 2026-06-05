import { describe, it, expect } from 'vitest';
import config from './playwright-genai.config.js';

describe('playwright-genai.config', () => {
  it('should export config', () => {
    expect(config).toBeDefined();
  });
});
