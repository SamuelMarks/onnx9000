import { describe, it, expect } from "vitest";
import { extractLlamaMetadata } from "../src/llama.js";
import { Graph } from "@onnx9000/core";

describe("llama", () => {
  it("should extract metadata", () => {
    const g = new Graph("llama");
    g.tensors["embed_tokens.weight"] = {
      shape: [100, 100],
      dtype: "float32",
    } as any;
    const meta = extractLlamaMetadata(g);
    expect(meta["llama.vocab_size"]).toBe(100);
  });
});
