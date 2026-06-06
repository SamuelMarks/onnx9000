import { describe, it, expect } from "vitest";
import { Onnxchecker } from "../src/index.js";

describe("Onnxchecker", () => {
  it("should run", () => {
    expect(new Onnxchecker().run()).toBeDefined();
  });
});
