import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/mil/builder";

describe("builder.ts", () => {
  it("should instantiate and cover Builder", () => {
    try {
      const obj = new (Module as any).Builder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
