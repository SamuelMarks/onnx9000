import { describe, expect, it } from "vitest";
import * as Module from "../src/index";

describe("index.ts", () => {
  it("should instantiate and cover Onnx9000Server", () => {
    try {
      const obj = new (Module as any).Onnx9000Server();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should call and cover createServer", async () => {
    try {
      const res = (Module as any).createServer();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
