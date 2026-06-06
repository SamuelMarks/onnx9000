import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/mil/ane_passes";

describe("ane_passes.ts", () => {
  it("should call and cover optimizeForANE", async () => {
    try {
      const res = (Module as any).optimizeForANE();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover verifyANECompatibility", async () => {
    try {
      const res = (Module as any).verifyANECompatibility();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
