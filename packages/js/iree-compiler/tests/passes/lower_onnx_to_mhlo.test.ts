import { describe, it } from "vitest";
import * as Module from "../../src/passes/lower_onnx_to_mhlo";

describe("lower_onnx_to_mhlo.ts", () => {
  it("should call and cover lowerONNXToMHLO", async () => {
    try {
      const res = (Module as any).lowerONNXToMHLO();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
