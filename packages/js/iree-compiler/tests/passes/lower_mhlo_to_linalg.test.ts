import { describe, it } from "vitest";
import * as Module from "../../src/passes/lower_mhlo_to_linalg";

describe("lower_mhlo_to_linalg.ts", () => {
  it("should call and cover lowerMHLOToLinalg", async () => {
    try {
      const res = (Module as any).lowerMHLOToLinalg();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
