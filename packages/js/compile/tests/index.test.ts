import { describe, it, expect } from "vitest";
import { Compile } from "../src/index.js";

describe("Compile", () => {
  it("should run", () => {
    const c = new Compile();
    expect(c.run()).toBe("[compile] processed");
  });
});
