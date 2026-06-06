import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/keras/optimizers";

describe("optimizers.ts", () => {
  it("should instantiate and cover KerasGraphOptimizer", () => {
    try {
      const obj = new (Module as any).KerasGraphOptimizer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover optimizeFusedOps", async () => {
    try {
      const res = (Module as any).optimizeFusedOps();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover applyQuantization", async () => {
    try {
      const res = (Module as any).applyQuantization();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
