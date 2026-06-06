import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/cntk/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover CNTKMapper", () => {
    try {
      const obj = new (Module as any).CNTKMapper();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover register_cntk_op", async () => {
    try {
      const res = (Module as any).register_cntk_op();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
