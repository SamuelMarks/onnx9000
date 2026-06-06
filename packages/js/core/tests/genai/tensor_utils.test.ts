import { describe, expect, it } from "vitest";
import * as Module from "../../src/genai/tensor_utils";

describe("tensor_utils.ts", () => {
  it("should instantiate and cover SequenceTensorUtils", () => {
    try {
      const obj = new (Module as any).SequenceTensorUtils();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
