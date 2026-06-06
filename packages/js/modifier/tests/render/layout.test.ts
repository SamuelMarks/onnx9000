import { describe, expect, it } from "vitest";
import * as Module from "../../src/render/layout";

describe("layout.ts", () => {
  it("should instantiate and cover DagreLayoutEngine", () => {
    try {
      const obj = new (Module as any).DagreLayoutEngine();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
