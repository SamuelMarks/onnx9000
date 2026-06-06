import { describe, it, expect } from "vitest";
import { Hummingbird } from "../src/index.js";

describe("Hummingbird", () => {
  it("should run", () => {
    expect(new Hummingbird().run()).toBeDefined();
  });
});
