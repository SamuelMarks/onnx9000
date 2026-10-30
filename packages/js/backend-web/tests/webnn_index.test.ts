import { describe, expect, it } from 'vitest';
import { WebNNProvider } from '../src/providers/webnn/index.js';

describe('WebNNProvider', () => {
  it('should init', async () => {
    const prov = new WebNNProvider();
    const g = globalThis as typeof globalThis & {
      navigator: unknown;
      MLGraphBuilder: unknown;
    };
    g.navigator = { ml: { createContext: async () => ({}) } };
    g.MLGraphBuilder = class {};
    await prov.initialize();
    const provInternal = prov as never as {
      contextManager: { getContext(): unknown };
    };
    expect(provInternal.contextManager.getContext()).toBeDefined();
  });
});
