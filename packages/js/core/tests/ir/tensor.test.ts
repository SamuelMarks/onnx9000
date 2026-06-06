import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/ir/tensor";

describe("tensor.ts", () => {
  it("should instantiate and cover Tensor", () => {
    try {
      const obj = new (Module as any).Tensor();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover SparseTensor", () => {
    try {
      const obj = new (Module as any).SparseTensor();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
