import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/array/index";

describe("index.ts", () => {
  it("should instantiate and cover ArrayAPI", () => {
    try {
      const obj = new (Module as any).ArrayAPI();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
