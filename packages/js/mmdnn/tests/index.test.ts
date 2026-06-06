import { describe, it, expect } from "vitest";
import { Mmdnn } from "../src/index.js";

describe("Mmdnn", () => {
  it("should run", () => {
    expect(new Mmdnn().run()).toBeDefined();
  });
});
