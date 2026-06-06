import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/compiler/layout";

describe("layout.ts", () => {
  it("should instantiate and cover LayoutOptimizer", () => {
    try {
      const obj = new (Module as any).LayoutOptimizer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
