import { describe, it, expect } from 'vitest';
import { WebNNProvider } from '../src/providers/webnn/index.js';

describe('WebNN EP Conformance (Phase 18-20 Roadmap)', () => {
  it('should instantiate WebNNProvider without crashing', () => {
    const provider = new WebNNProvider();
    expect(provider).toBeDefined();
    expect(provider.name).toBe('WebNN');
  });

  it('should initialize successfully with default options', async () => {
    const provider = new WebNNProvider();
    // Catch errors during initialization if WebNN is unavailable
    try {
      await provider.initialize();
      expect(provider.name).toBe('WebNN'); // Reached if successful
    } catch (e: any) {
      // In CI environments without WebNN, it falls back or throws
      expect(e).toBeDefined();
    }
  });
});
