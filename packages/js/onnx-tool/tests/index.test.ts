import { describe, it, expect } from "vitest";
import { ONNXTool } from "../src/index.js";

describe("ONNXTool", () => {
  it("should run", () => {
    expect(new ONNXTool().process("test")).toContain("test");
  });
});
