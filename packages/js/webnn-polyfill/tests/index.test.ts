import { describe, expect, it } from "vitest";
import { PolyfillML } from "../src/index.js";

describe("WebNN polyfill", () => {
  it("should inject into navigator", async () => {
    if (typeof window !== "undefined") {
      expect((window.navigator as any).ml).toBeDefined();
      expect((window as any).MLContext).toBeDefined();

      const ml = new PolyfillML();
      const ctx = await ml.createContext();
      expect(ctx).toBeDefined();
    }
  });
});
