import { describe, expect, it } from "vitest";
import { Gguf } from "../src/index.js";

describe("Gguf", () => {
  it("should run", () => {
    expect(new Gguf().run()).toBeDefined();
  });
});
