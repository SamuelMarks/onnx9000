import { describe, it, expect, vi } from "vitest";
import { compileOnnxToC, initCompiler } from "../src/index.js";

vi.mock("@onnx9000/core", () => ({
  parseModelProto: vi.fn().mockResolvedValue({
    name: "test",
    inputs: [],
    outputs: [],
    tensors: {},
    initializers: [],
    nodes: [],
  }),
  BufferReader: class {},
}));

describe("c-compiler index", () => {
  it("should compile", async () => {
    const init = await initCompiler();
    expect(init.initialized).toBe(true);

    const res = await compileOnnxToC(new Uint8Array());
    expect(res.header).toBeDefined();
    expect(res.source).toBeDefined();
  });
});
