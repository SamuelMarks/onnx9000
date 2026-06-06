import { describe, expect, it } from "vitest";
import { SKL2ONNXConverter } from "../src/index.js";

describe("SKL2ONNXConverter", () => {
  it("should convert", () => {
    const c = new SKL2ONNXConverter();
    expect(c.convert("model")).toContain("[ONNX-IR]");
    expect(() => c.convert("")).toThrow();
  });
});
