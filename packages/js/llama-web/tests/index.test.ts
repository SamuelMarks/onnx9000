import { describe, expect, it } from "vitest";
import { LlamaWeb } from "../src/index.js";

describe("LlamaWeb", () => {
  it("should run", () => {
    const l = new LlamaWeb();
    expect(l.run("test")).toBeDefined();
    expect(() => l.run("")).toThrow();
  });
});
