import { describe, expect, it } from "vitest";
import { llama7b } from "../src/models/llama.js";

describe("LLaMA", () => {
  it("should create and call", () => {
    const model = llama7b();
    expect(model).toBeDefined();
    const out = model.call({} as any, {} as any);
    expect(out).toBeDefined();
  });
});
