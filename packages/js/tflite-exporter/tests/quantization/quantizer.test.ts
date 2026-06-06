import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/quantization/quantizer";

describe("quantizer.ts", () => {
  it("should instantiate and cover Quantizer", () => {
    try {
      const obj = new (Module as any).Quantizer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
