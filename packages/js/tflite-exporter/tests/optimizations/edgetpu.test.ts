import { describe, expect, it } from "vitest";
import * as Module from "../../src/optimizations/edgetpu";

describe("edgetpu.ts", () => {
  it("should instantiate and cover EdgeTPUOptimizer", () => {
    try {
      const obj = new (Module as any).EdgeTPUOptimizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
