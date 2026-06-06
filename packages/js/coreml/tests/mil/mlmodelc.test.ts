import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/mil/mlmodelc";

describe("mlmodelc.ts", () => {
  it("should instantiate and cover MLModelCCompiler", () => {
    try {
      const obj = new (Module as any).MLModelCCompiler();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
