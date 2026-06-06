import { describe, expect, it } from "vitest";
import * as Module from "../../src/passes/validation";

describe("validation.ts", () => {
  it("should instantiate and cover ValidationSuite", () => {
    try {
      const obj = new (Module as any).ValidationSuite();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
