import { describe, it, expect } from "vitest";
import { ZeroDepClassifier } from "../src/index.js";

describe("ZeroDepClassifier", () => {
  it("processes", () => {
    expect(new ZeroDepClassifier().process("test")).toBe(
      "Zero Dep Classifier processed test",
    );
  });
});
