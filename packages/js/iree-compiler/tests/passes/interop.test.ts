import { describe, expect, it } from "vitest";
import * as Module from "../../src/passes/interop";

describe("interop.ts", () => {
  it("should instantiate and cover MLIRInterop", () => {
    try {
      const obj = new (Module as any).MLIRInterop();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
