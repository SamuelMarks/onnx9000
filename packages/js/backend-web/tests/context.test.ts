import { describe, expect, it } from 'vitest';
import { WebNNContextManager } from '../src/providers/webnn/context.js';

describe('WebNNContextManager', () => {
  it('should initialize', async () => {
    const mgr = WebNNContextManager.getInstance();
    const g = globalThis as typeof globalThis & {
      navigator: unknown;
      MLGraphBuilder: unknown;
    };
    g.navigator = { ml: { createContext: async () => ({}) } };
    g.MLGraphBuilder = class {};

    await mgr.initialize();
    expect(mgr.getContext()).toBeDefined();
    expect(mgr.getBuilder()).toBeDefined();
  });
});
