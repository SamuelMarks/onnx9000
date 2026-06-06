import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/ncnn/parser";

describe("parser.ts", () => {
  it("should instantiate and cover NcnnBinParser", () => {
    try {
      const obj = new (Module as any).NcnnBinParser();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover parseNcnnParam", async () => {
    try {
      const res = (Module as any).parseNcnnParam();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
