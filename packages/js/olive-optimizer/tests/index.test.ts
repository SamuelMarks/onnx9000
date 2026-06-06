import { describe, expect, it } from "vitest";
import { OliveOptimizer } from "../src/index.js";

describe("OliveOptimizer", () => {
  it("should run", () => {
    expect(new OliveOptimizer().process("test")).toContain("test");
  });
});
