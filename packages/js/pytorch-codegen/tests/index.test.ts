import { describe, expect, it } from "vitest";
import { Pytorchcodegen } from "../src/index.js";

describe("Pytorchcodegen", () => {
  it("should run", () => {
    expect(new Pytorchcodegen().run()).toBeDefined();
  });
});
