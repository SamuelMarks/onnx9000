import { describe, expect, it } from "vitest";
import { Simplify } from "../src/index.js";

describe("Simplify", () => {
  it("should run", () => {
    expect(new Simplify().run()).toBeDefined();
  });
});
