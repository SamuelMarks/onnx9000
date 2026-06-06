import { describe, it, expect, vi } from "vitest";
import { DiffusionPipeline } from "../src/pipeline.js";

vi.mock("../src/utils.js", () => ({
  PyTorchPCG: class {
    nextFloat() {
      return 0.5;
    }
  },
  parseModelIndex: vi.fn().mockResolvedValue({}),
}));

describe("DiffusionPipeline", () => {
  it("should create and call", async () => {
    const p = await DiffusionPipeline.fromPretrained("test");
    expect(p).toBeDefined();
    const res = await p.call("prompt", 2);
    expect(res).toBeDefined();

    p.freeMemory();
  });
});
