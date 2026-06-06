import { describe, expect, it } from "vitest";
import { registry } from "../src/index.js";

describe("custom-ops", () => {
  it("should register", () => {
    registry.register("test", () => {});
    expect(registry.getOp("test")).toBeDefined();
    expect(registry.listOps()).toContain("test");
  });
});
