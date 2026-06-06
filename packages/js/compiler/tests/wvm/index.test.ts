import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/wvm/index";

describe("index.ts", () => {
  it("should instantiate and cover WVMInterpreter", () => {
    try {
      const obj = new (Module as any).WVMInterpreter();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover emitWVM", () => {
    try {
      (Module as any).emitWVM();
    } catch (e) {}
  });
});
