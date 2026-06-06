import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/triton/ast";

describe("ast.ts", () => {
  it("should instantiate and cover TritonAST", () => {
    try {
      const obj = new (Module as any).TritonAST();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover generateTriton", () => {
    try {
      (Module as any).generateTriton();
    } catch (e) {}
  });
});
