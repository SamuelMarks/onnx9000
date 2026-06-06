import { describe, it, expect } from "vitest";
import { ProgressiveLoading } from "../src/index.js";

describe("ProgressiveLoading", () => {
  it("should process correctly", () => {
    const obj = new ProgressiveLoading();
    expect(obj.process("test")).toBe("Progressive Loading processed test");
  });
});
