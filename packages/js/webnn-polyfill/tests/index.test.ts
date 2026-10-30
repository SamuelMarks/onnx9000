import { describe, expect, it } from 'vitest';
import { PolyfillML } from '../src/index.js';

describe('WebNN polyfill', () => {
  it('should inject into navigator', async () => {
    if (typeof window !== 'undefined') {
      expect((window.navigator as Navigator & { ml?: unknown }).ml).toBeDefined();
      expect((window as Window & { MLContext?: unknown }).MLContext).toBeDefined();

      const ml = new PolyfillML();
      const ctx = await ml.createContext();
      expect(ctx).toBeDefined();
    }
  });
});
