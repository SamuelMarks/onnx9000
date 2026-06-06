import { describe, expect, it } from "vitest";
import { Onnxchecker } from "../src/index.js";

describe("Onnxchecker", () => {
  it("should run", () => {
    expect(new Onnxchecker().run()).toBeDefined();
  });
});
