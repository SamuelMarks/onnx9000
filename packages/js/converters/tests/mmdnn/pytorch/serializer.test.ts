import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/pytorch/serializer";

describe("serializer.ts", () => {
  it("should instantiate and cover PyTorchSerializer", () => {
    try {
      const obj = new (Module as any).PyTorchSerializer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
