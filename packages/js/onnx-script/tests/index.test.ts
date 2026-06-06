import { describe, expect, it } from "vitest";
import { Onnxscript } from "../src/index.js";

describe("Onnxscript", () => {
  it("should run", () => {
    expect(new Onnxscript().run()).toBeDefined();
  });
});
