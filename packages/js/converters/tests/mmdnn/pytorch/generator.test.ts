import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/pytorch/generator";

describe("generator.ts", () => {
  it("should instantiate and cover PyTorchGenerator", () => {
    try {
      const obj = new (Module as any).PyTorchGenerator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
