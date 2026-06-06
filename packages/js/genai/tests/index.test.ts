import { describe, expect, it } from "vitest";
import { Genai } from "../src/index.js";

describe("Genai", () => {
  it("should run", () => {
    expect(new Genai().run()).toBeDefined();
  });
});
