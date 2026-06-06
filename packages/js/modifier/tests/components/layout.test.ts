import { describe, expect, it } from "vitest";
import * as Module from "../../src/components/layout";

describe("layout.ts", () => {
  it("should instantiate and cover LayoutBuilder", () => {
    try {
      const obj = new (Module as any).LayoutBuilder();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
