import { describe, it, expect } from "vitest";
import { ZeroDepClassifier } from "../src/index.js";

describe("ZeroDepClassifier", () => {
  it("should run", () => {
    expect(new ZeroDepClassifier().process("test")).toContain("test");
  });
});
