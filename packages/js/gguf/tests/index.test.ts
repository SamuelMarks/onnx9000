import { describe, it, expect } from "vitest";
import { Gguf } from "../src/index.js";

describe("Gguf", () => {
  it("should run", () => {
    expect(new Gguf().run()).toBeDefined();
  });
});
