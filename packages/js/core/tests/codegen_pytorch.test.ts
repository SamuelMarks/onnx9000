import { describe, it, expect } from "vitest";
import { ONNXToPyTorchVisitor } from "../src/codegen/pytorch.js";

describe("ONNXToPyTorchVisitor", () => {
  it("should generate code", () => {
    const v = new ONNXToPyTorchVisitor({
      name: "test",
      inputs: [],
      outputs: [],
      tensors: {},
      initializers: [],
      nodes: [],
    } as any);
    const code = v.generate();
    expect(code).toContain("import torch");
  });
});
