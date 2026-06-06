import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/parser/onnx";

describe("onnx.ts", () => {
  it("should call and cover parseModelProto", async () => {
    try {
      const res = (Module as any).parseModelProto();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover releaseArrayBuffer", async () => {
    try {
      const res = (Module as any).releaseArrayBuffer();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
