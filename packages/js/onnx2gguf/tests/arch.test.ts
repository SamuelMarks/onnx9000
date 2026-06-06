import { Graph } from "@onnx9000/core";
import { describe, expect, it } from "vitest";
import { extractMetadata, inferArchitecture } from "../src/arch.js";

describe("arch", () => {
  it("should infer", () => {
    const g = new Graph("mistral");
    expect(inferArchitecture(g)).toBe("mistral");
  });

  it("should extract metadata", () => {
    const g = new Graph("llama");
    const res = extractMetadata(g);
    expect(res).toBeDefined();
  });
});
