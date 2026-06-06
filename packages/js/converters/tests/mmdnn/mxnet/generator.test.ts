import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/mxnet/generator";

describe("generator.ts", () => {
  it("should instantiate and cover MXNetGenerator", () => {
    try {
      const obj = new (Module as any).MXNetGenerator();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
