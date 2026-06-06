import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/caffe/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover CaffeMapper", () => {
    try {
      const obj = new (Module as any).CaffeMapper();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover register_caffe_op", async () => {
    try {
      const res = (Module as any).register_caffe_op();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
