import { describe, it, expect } from "vitest";
import { Mlir } from "../src/index.js";

describe("Mlir", () => {
  it("should run", () => {
    expect(new Mlir().run()).toBeDefined();
  });
});
