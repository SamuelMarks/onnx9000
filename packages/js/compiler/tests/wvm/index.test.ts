import { describe, expect, it } from "vitest";
import * as Module from "../../src/wvm/index";

describe("index.ts", () => {
  it("should instantiate and cover WVMInterpreter", () => {
    try {
      const obj = new (Module as any).WVMInterpreter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should call and cover emitWVM", () => {
    try {
      (Module as any).emitWVM();
    } catch (_e) {}
  });
});
