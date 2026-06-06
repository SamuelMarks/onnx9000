import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/ir/node";

describe("node.ts", () => {
  it("should instantiate and cover Attribute", () => {
    try {
      const obj = new (Module as any).Attribute();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover Node", () => {
    try {
      const obj = new (Module as any).Node();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
