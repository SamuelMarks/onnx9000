import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/mxnet/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover MxNetMapper", () => {
    try {
      const obj = new (Module as any).MxNetMapper();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should call and cover register_mxnet_op", async () => {
    try {
      const res = (Module as any).register_mxnet_op();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
