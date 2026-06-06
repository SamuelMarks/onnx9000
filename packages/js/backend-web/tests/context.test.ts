import { describe, expect, it } from 'vitest';
import { WebNNContextManager } from '../src/providers/webnn/context.js';

describe('WebNNContextManager', () => {
  it('should initialize', async () => {
    const mgr = WebNNContextManager.getInstance();
    (globalThis as any).navigator = { ml: { createContext: async () => ({}) } };
    (globalThis as any).MLGraphBuilder = class {};

    await mgr.initialize();
    expect(mgr.getContext()).toBeDefined();
    expect(mgr.getBuilder()).toBeDefined();
  });
});
