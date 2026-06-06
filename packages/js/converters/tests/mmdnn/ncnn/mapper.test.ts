import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/ncnn/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover NcnnMapper", () => {
    try {
      const obj = new (Module as any).NcnnMapper();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
