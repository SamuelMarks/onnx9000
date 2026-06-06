import { describe, it, expect } from "vitest";
import { WebNNProvider } from "../src/providers/webnn/index.js";

describe("WebNNProvider", () => {
  it("should init", async () => {
    const prov = new WebNNProvider();
    (globalThis as any).navigator = { ml: { createContext: async () => ({}) } };
    (globalThis as any).MLGraphBuilder = class {};
    await prov.initialize();
    expect((prov as any).contextManager.getContext()).toBeDefined();
  });
});
