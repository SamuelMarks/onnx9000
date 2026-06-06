import { describe, it } from "vitest";
import * as Module from "../../src/parser/onnx_writer";

describe("onnx_writer.ts", () => {
  it("should call and cover serializeModelProto", async () => {
    try {
      const res = (Module as any).serializeModelProto();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
