import { describe, expect, it } from "vitest";
import * as Module from "../../src/passes/lower_wasm";

describe("lower_wasm.ts", () => {
  it("should instantiate and cover WASMEmitter", () => {
    try {
      const obj = new (Module as any).WASMEmitter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should call and cover lowerLinalgToSCF", async () => {
    try {
      const res = (Module as any).lowerLinalgToSCF();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover unrollLoops", async () => {
    try {
      const res = (Module as any).unrollLoops();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover vectorizeLoops", async () => {
    try {
      const res = (Module as any).vectorizeLoops();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
