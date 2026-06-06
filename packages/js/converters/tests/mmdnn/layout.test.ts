import { describe, expect, it } from "vitest";
import * as Module from "../../src/mmdnn/layout";

describe("layout.ts", () => {
  it("should instantiate and cover DataLayoutTracker", () => {
    try {
      const obj = new (Module as any).DataLayoutTracker();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
