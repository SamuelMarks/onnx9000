import { describe, it, expect } from "vitest";
import { extractTokenizerMetadata } from "../src/tokenizer.js";

describe("tokenizer", () => {
  it("should extract metadata", () => {
    const meta = extractTokenizerMetadata(
      '{"model": {"type": "BPE", "vocab": {"a": 1}}}',
      0,
    );
    expect(meta["tokenizer.ggml.model"]).toBe("gpt2");

    const emptyMeta = extractTokenizerMetadata(null, 10);
    expect(emptyMeta["tokenizer.ggml.tokens"].length).toBe(10);
  });
});
